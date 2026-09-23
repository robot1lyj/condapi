"""Selective TorchAO FP8 experiment; synthetic cases, never a policy acceptance test."""

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
import torchao
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerRow

logging.basicConfig(level=logging.INFO)
out = Path("/reports")
recipe = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
report = {
    "scope": "three_synthetic_inputs_not_task_accuracy",
    "torchao": torchao.__version__,
    "recipe": str(recipe),
    "results": [],
}


def save():
    (out / "quant-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


torch.manual_seed(42)
probe = torch.nn.Sequential(torch.nn.Linear(3072, 4096, device="cuda", dtype=torch.bfloat16)).eval()
quantize_(probe, recipe)
x = torch.randn(64, 3072, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    y = probe(x)
torch.cuda.synchronize()
report["kernel_probe"] = {
    "finite": bool(torch.isfinite(y).all()),
    "weight_type": type(probe[0].weight).__name__,
    "operators": sorted(
        {event.key for event in prof.key_averages() if "scaled_mm" in event.key or "float8" in event.key}
    ),
}
save()
print(json.dumps(report["kernel_probe"]), flush=True)
del probe, x, y
torch.cuda.empty_cache()
cfg = OmegaConf.load("/opt/openwam/configs/deploy.yaml")
cfg.optimization.compile.enabled = False
cfg.optimization.dit_cache.enabled = False
cfg.optimization.prompt_embed_cache.enabled = True
cfg.optimization.decode_video = False
start = time.perf_counter()
server = build_server_from_config(cfg, "/checkpoint", device="cuda")
report["load_seconds"] = time.perf_counter() - start
engine = server.engine
preprocess = ObsPreprocessor.from_cfg(server.cfg, engine)
cases = []
for seed in [42, 43, 44]:
    rng = np.random.default_rng(seed)
    images = [Image.fromarray(rng.integers(0, 256, (384, 320, 3), dtype=np.uint8)) for _ in range(3)]
    state = np.zeros(20, dtype=np.float32)
    state[3:9] = state[13:19] = [1, 0, 0, 0, 1, 0]
    obs = preprocess.preprocess(
        {
            "images": dict(zip(["head_camera", "left_wrist_camera", "right_wrist_camera"], images, strict=True)),
            "state": state,
            "prompt": "Pick up the object and place it in the tray.",
        }
    )
    cases.append({"first_frame_image": [obs["image"]], "proprio": state, "prompt": obs["prompt"], "seed": seed})
blocks = engine.architecture.video_backbone.dit.blocks
targets = {name: module for name, module in blocks.named_modules() if isinstance(module, torch.nn.Linear)}
report["eligible_video_linears"] = list(targets)
reference = None
for mode in ["bf16", "fp8_ffn", "fp8_blocks", "fp8_blocks_compile"]:
    if mode == "fp8_ffn":
        quantize_(
            blocks, recipe, filter_fn=lambda module, name: isinstance(module, torch.nn.Linear) and ".ffn." in name
        )
    if mode == "fp8_blocks":
        quantize_(
            blocks, recipe, filter_fn=lambda module, name: isinstance(module, torch.nn.Linear) and ".ffn." not in name
        )
    if mode == "fp8_blocks_compile":
        cfg.optimization.compile.enabled = True
        engine.architecture.apply_compile_optimizations(cfg.optimization.compile)
    row = {"mode": mode, "weight_types": {name: type(module.weight).__name__ for name, module in targets.items()}}
    torch.cuda.reset_peak_memory_stats()
    measurements = []
    outputs = []
    for index in range(10):
        case_index = 0 if index == 0 else (index - 1) // 3
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = engine.generate(cases[case_index])
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        a = result["actions"]
        if isinstance(a, torch.Tensor):
            a = a.detach().float().cpu().numpy()
        a = np.asarray(a)
        if a.shape != (32, 20) or not np.isfinite(a).all():
            raise ValueError(f"Invalid actions {a.shape}")
        if index == 0:
            row["cold_ms"] = ms
        else:
            measurements.append(ms)
            if index % 3 == 0:
                outputs.append(a.copy())
        print(json.dumps({"mode": mode, "index": index, "ms": ms}), flush=True)
    outputs = np.stack(outputs)
    np.save(out / f"{mode}-actions.npy", outputs)
    if reference is None:
        reference = outputs.copy()
    diff = np.abs(outputs - reference)
    row.update(
        {
            "times_ms": measurements,
            "p50_ms": float(np.median(measurements)),
            "p95_ms": float(np.percentile(measurements, 95)),
            "max_abs": float(diff.max()),
            "mae": float(diff.mean()),
            "position_max": float(diff[..., [0, 1, 2, 10, 11, 12]].max()),
            "rot6d_max": float(diff[..., [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]].max()),
            "gripper_max": float(diff[..., [9, 19]].max()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    report["results"].append(row)
    save()
print("QUANT_BENCHMARK_COMPLETED", flush=True)
