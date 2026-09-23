"""Probe actual OpenWAM-Alpha joint-attention mask and exact operator splitting.

Thor-only diagnostic. One synthetic observation loads the pinned official model;
the operator microbenchmark reuses Q/K/V from its first joint-attention layer.
It does not modify model files or claim robot-task accuracy.
"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.server import build_server_from_config
from PIL import Image
import torch
from torch.nn import functional
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel


def event_times(call, *, repeats=60):
    for _ in range(8):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {"p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    config = OmegaConf.load("/opt/openwam/configs/deploy.yaml")
    config.optimization.compile.enabled = False
    config.optimization.dit_cache.enabled = False
    config.optimization.prompt_embed_cache.enabled = False
    config.optimization.decode_video = False
    server = build_server_from_config(config, "/checkpoint", device="cuda")
    engine = server.engine
    driver = engine.architecture.mot_driver
    captured = {}
    original_step = driver.step
    original_mixed = driver._mixed_attention  # noqa: SLF001

    def step_capture(layer_id, vstate, astate, **kwargs):
        if layer_id == 0 and not captured:
            captured["prefix_tokens"] = driver._video_tokens_per_frame(vstate)  # noqa: SLF001
            captured["video_tokens"] = vstate.hidden_states.shape[1]
            captured["time_mod_shape"] = list(vstate.time_mod.shape)
            captured["rope_freqs_shape"] = list(vstate.rope_freqs.shape)
            captured["grid"] = [vstate.grid_frames, vstate.grid_height, vstate.grid_width]
            captured["vace_hints"] = len(vstate.vace_hints or [])
        return original_step(layer_id, vstate, astate, **kwargs)

    def mixed_capture(q_cat, k_cat, v_cat, mask):
        if "q_cat" not in captured:
            captured["q_cat"] = q_cat.detach().clone()
            captured["k_cat"] = k_cat.detach().clone()
            captured["v_cat"] = v_cat.detach().clone()
            captured["mask"] = mask.detach().clone()
        return original_mixed(q_cat, k_cat, v_cat, mask)

    driver.step = step_capture
    driver._mixed_attention = mixed_capture  # noqa: SLF001
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
    driver._mixed_attention = original_mixed  # noqa: SLF001
    actions = output["actions"]
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().float().cpu().numpy()
    if np.asarray(actions).shape != (32, 20) or not np.isfinite(actions).all():
        raise ValueError("unexpected or nonfinite actions")

    mask = captured["mask"]
    if mask.ndim != 2:
        raise ValueError(f"expected shared 2D mask, got {tuple(mask.shape)}")
    prefix = captured["prefix_tokens"]
    total = mask.shape[0]
    geometry = {
        "prefix_allows_only_prefix": bool(mask[:prefix, :prefix].all() and not mask[:prefix, prefix:].any()),
        "suffix_allows_all_keys": bool(mask[prefix:, :].all()),
        "counts": [int(x) for x in torch.unique(mask.sum(-1)).tolist()],
    }
    report = {
        "scope": "actual_first_joint_layer_qkv_synthetic_observation_operator_only",
        "shape": {
            key: captured[key]
            for key in ("prefix_tokens", "video_tokens", "time_mod_shape", "rope_freqs_shape", "grid", "vace_hints")
        },
        "mask": {"shape": list(mask.shape), **geometry},
    }
    if not all((geometry["prefix_allows_only_prefix"], geometry["suffix_allows_all_keys"])):
        Path("/reports/segmented-attention.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"mask_not_segmentable": geometry}), flush=True)
        return
    q = captured["q_cat"].reshape(1, total, driver.num_heads, driver.head_dim).transpose(1, 2)
    k = captured["k_cat"].reshape(1, total, driver.num_heads, driver.head_dim).transpose(1, 2)
    v = captured["v_cat"].reshape(1, total, driver.num_heads, driver.head_dim).transpose(1, 2)

    def masked():
        return functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def segmented(backend):
        if backend is None:
            head = functional.scaled_dot_product_attention(q[:, :, :prefix], k[:, :, :prefix], v[:, :, :prefix])
            tail = functional.scaled_dot_product_attention(q[:, :, prefix:], k, v)
            return torch.cat((head, tail), dim=2)
        with sdpa_kernel(backend):
            head = functional.scaled_dot_product_attention(q[:, :, :prefix], k[:, :, :prefix], v[:, :, :prefix])
            tail = functional.scaled_dot_product_attention(q[:, :, prefix:], k, v)
        return torch.cat((head, tail), dim=2)

    reference = masked().float()
    report["qkv_shape"] = list(q.shape)
    report["operators"] = []
    for name, call in (
        ("masked_auto", masked),
        ("split_flash", lambda: segmented(SDPBackend.FLASH_ATTENTION)),
        ("split_auto", lambda: segmented(None)),
    ):
        try:
            with torch.inference_mode():
                result = call().float()
                delta = (result - reference).abs()
                row = {
                    "name": name,
                    **event_times(call),
                    "max_abs_vs_masked": float(delta.max()),
                    "mae_vs_masked": float(delta.mean()),
                }
        except Exception as exc:
            row = {"name": name, "error": str(exc)}
        report["operators"].append(row)
        print(json.dumps(row), flush=True)
    Path("/reports/segmented-attention.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
