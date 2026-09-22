"""Thor-only OpenWAM smoke/performance comparison; synthetic inputs, no robot."""
# ruff: noqa: SLF001 -- pinned upstream engine has no runtime cache-switch public API.

import json
import logging
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.server import build_server_from_config
from PIL import Image
import torch

out = Path("/reports")
logging.basicConfig(level=logging.INFO)
cfg = OmegaConf.load("/opt/openwam/configs/deploy.yaml")
cfg.optimization.compile.enabled = False
cfg.optimization.dit_cache.enabled = False
cfg.optimization.prompt_embed_cache.enabled = False
cfg.optimization.decode_video = False
print("LOADING_OFFICIAL_CHECKPOINT", flush=True)
start = time.perf_counter()
server = build_server_from_config(cfg, "/checkpoint", device="cuda")
engine = server.engine
print(f"MODEL_LOADED {time.perf_counter() - start:.3f}s", flush=True)
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
conditions = {
    "first_frame_image": [obs["image"]],
    "proprio": state,
    "prompt": obs["prompt"],
    "seed": 42,
}
report = {"scope": "synthetic_smoke_not_task_accuracy", "results": []}
reference = None
for name in ["baseline", "prompt_cache", "dit_cache", "compile"]:
    if name == "prompt_cache":
        from openwam.deploy.engine import _BoundedPromptEmbedCache

        engine._prompt_embed_cache = _BoundedPromptEmbedCache(32)
    if name == "dit_cache":
        from openwam.deploy.optimizations import DiTVelocityCache

        engine._dit_cache = DiTVelocityCache(cosine_threshold=0.99, max_consecutive_skips=3)
    if name == "compile":
        engine._dit_cache = None
        cfg.optimization.compile.enabled = True
        engine.architecture.apply_compile_optimizations(cfg.optimization.compile)
    elapsed = []
    for repeat in range(4):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = engine.generate(conditions)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        actions = result["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions)
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite actions")
        print(json.dumps({"mode": name, "repeat": repeat, "ms": ms, "shape": actions.shape}), flush=True)
        elapsed.append(ms)
    if reference is None:
        reference = actions.copy()
    np.save(out / f"{name}-actions.npy", actions)
    row = {
        "mode": name,
        "cold_ms": elapsed[0],
        "times_ms": elapsed[1:],
        "p50_ms": float(np.median(elapsed[1:])),
        "p95_ms": float(np.percentile(elapsed[1:], 95)),
        "shape": list(actions.shape),
        "max_abs_vs_baseline": float(np.max(np.abs(actions - reference))),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    if engine._dit_cache is not None:
        row["cache_stats"] = engine._dit_cache.stats
    report["results"].append(row)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
print("BENCHMARK_COMPLETED", flush=True)
