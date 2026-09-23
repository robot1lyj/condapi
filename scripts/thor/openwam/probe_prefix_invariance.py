"""Check whether Alpha's clean first-frame state repeats across denoising passes.

Diagnostic only: reads an existing checkpoint, runs eager BF16 once, and leaves
the model unmodified on disk. The model must be the pinned TI2V mutual variant.
"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.server import build_server_from_config
from PIL import Image
import torch


def main() -> None:
    config = OmegaConf.load("/opt/openwam/configs/deploy.yaml")
    config.optimization.compile.enabled = False
    config.optimization.dit_cache.enabled = False
    config.optimization.prompt_embed_cache.enabled = False
    config.optimization.decode_video = False
    server = build_server_from_config(config, "/checkpoint", device="cuda")
    engine = server.engine
    driver = engine.architecture.mot_driver
    if driver is None:
        raise RuntimeError("MoT driver unavailable")

    snapshots = {}
    results = []
    pass_index = 0
    original_step = driver.step

    def capture(layer_id, vstate, astate, **kwargs):
        nonlocal pass_index
        if layer_id == 0:
            pass_index += 1
        if pass_index <= 2:
            prefix_tokens = driver._video_tokens_per_frame(vstate)  # noqa: SLF001
            if layer_id == 0:
                compare(-1, vstate.hidden_states[:, :prefix_tokens])
        vstate, astate = original_step(layer_id, vstate, astate, **kwargs)
        if pass_index <= 2:
            compare(layer_id, vstate.hidden_states[:, :prefix_tokens])
        return vstate, astate

    def compare(stage, value):
        current = value.detach().float().cpu()
        if pass_index == 1:
            snapshots[stage] = current
        elif pass_index == 2:
            difference = (current - snapshots[stage]).abs()
            results.append(
                {
                    "stage": stage,
                    "shape": list(current.shape),
                    "max_abs": float(difference.max()),
                    "mae": float(difference.mean()),
                }
            )

    driver.step = capture
    rng = np.random.default_rng(42)
    images = [Image.fromarray(rng.integers(0, 256, (384, 320, 3), dtype=np.uint8)) for _ in range(3)]
    state = np.zeros(20, dtype=np.float32)
    state[3:9] = state[13:19] = [1, 0, 0, 0, 1, 0]
    obs = ObsPreprocessor.from_cfg(server.cfg, engine).preprocess(
        {
            "images": dict(zip(["head_camera", "left_wrist_camera", "right_wrist_camera"], images, strict=True)),
            "state": state,
            "prompt": "Pick up the object and place it in the tray.",
        }
    )
    with torch.inference_mode():
        output = engine.generate(
            {
                "first_frame_image": [obs["image"]],
                "proprio": state,
                "prompt": obs["prompt"],
                "seed": 42,
                "denoise_steps": 10,
            }
        )
    driver.step = original_step
    actions = output["actions"]
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().float().cpu().numpy()
    if np.asarray(actions).shape != (32, 20) or not np.isfinite(actions).all():
        raise ValueError("Unexpected or nonfinite actions")
    if pass_index < 2 or len(results) != driver.num_layers + 1:
        raise RuntimeError(f"Incomplete capture: passes={pass_index}, stages={len(results)}")
    report = {
        "scope": "one_synthetic_observation_first_two_eager_forwards_only",
        "passes_seen": pass_index,
        "prefix_tokens": results[0]["shape"][1],
        "layers": driver.num_layers,
        "global_max_abs": max(row["max_abs"] for row in results),
        "results": results,
    }
    Path("/reports/prefix-invariance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passes_seen": pass_index, "global_max_abs": report["global_max_abs"]}), flush=True)


if __name__ == "__main__":
    main()
