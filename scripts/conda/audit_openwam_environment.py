"""Audit native OpenWAM dependencies; optional allocated GPU math, never a training loop."""

import argparse
from datetime import UTC
from datetime import datetime
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adapters.openwam.common import activate_source  # noqa: E402
from adapters.openwam.common import verify_source  # noqa: E402


def audit(*, gpu=False):
    if gpu and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU audit requires its own authorized Slurm allocation")
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled", OMP_NUM_THREADS="1")
    activate_source()
    revision = verify_source()
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    modules = [
        "torchvision",
        "av",
        "pyarrow",
        "h5py",
        "transformers",
        "diffusers",
        "deepspeed",
        "accelerate",
        "openwam.train.openwam_trainer",
        "openwam.deploy.model_loader",
        "openwam.deploy.engine",
        "adapters.openwam.data",
        "adapters.openwam.train",
        "adapters.openwam.infer",
    ]
    for name in modules:
        importlib.import_module(name)
    model = importlib.import_module("openwam.model")
    architectures = model.list_supported_architectures()
    if not architectures:
        raise ValueError("No native architectures registered")
    from accelerate import DeepSpeedPlugin  # noqa: PLC0415

    plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=1, offload_optimizer_device="none")
    if plugin.deepspeed_config["zero_optimization"]["stage"] != 2:
        raise ValueError("DeepSpeed ZeRO-2 configuration mismatch")
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "prefix": sys.prefix,
        "python": sys.version,
        "upstream_revision": revision,
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "imports": modules,
        "architectures": architectures,
        "versions": {
            name: version(name)
            for name in (
                "openwam",
                "torch",
                "torchvision",
                "transformers",
                "diffusers",
                "deepspeed",
                "accelerate",
                "numpy",
                "av",
                "hydra-core",
                "omegaconf",
                "safetensors",
            )
        },
        "cuda_build": torch.version.cuda,
        "scope": "dependency_import_and_config_only",
        "training_executed": False,
        "model_weights_loaded": False,
    }
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("Allocated CUDA device unavailable")
        with torch.no_grad():
            x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
            product = x @ x.T
            q = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
            attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            if not torch.isfinite(product).all() or not torch.isfinite(attention).all():
                raise RuntimeError("Nonfinite CUDA matrix/SDPA output")
            torch.cuda.synchronize()
        result.update(
            scope="allocated_gpu_math_and_dependency_audit",
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            gpu={
                "name": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "fp32_matmul": "passed",
                "bf16_sdpa": "passed",
            },
        )
    elif torch.cuda.is_initialized():
        raise RuntimeError("CPU audit unexpectedly initialized CUDA")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior evidence: output already exists")
    result = audit(gpu=args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
