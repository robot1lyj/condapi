"""Offline CPU environment check: no weights, GPU context or training job."""

import argparse
from datetime import UTC
from datetime import datetime
import importlib
from importlib.metadata import distributions
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys


def audit():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["OMP_NUM_THREADS"] = "1"
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    imports = [
        "torchvision",
        "torchcodec.decoders",
        "av",
        "transformers",
        "accelerate",
        "lerobot.scripts.lerobot_train",
        "lerobot.policies.evo1.modeling_evo1",
    ]
    for module in imports:
        importlib.import_module(module)
    types = importlib.import_module("lerobot.configs.types")
    config_cls = importlib.import_module("lerobot.policies.evo1.configuration_evo1").Evo1Config
    factory = importlib.import_module("lerobot.policies.evo1.processor_evo1").make_evo1_pre_post_processors
    cameras = [f"observation.images.{name}_rgb" for name in ("top", "left", "right")]
    cfg = config_cls(
        device="cpu",
        use_amp=False,
        binarize_gripper=False,
        input_features={
            "observation.state": types.PolicyFeature(type=types.FeatureType.STATE, shape=(14,)),
            **{key: types.PolicyFeature(type=types.FeatureType.VISUAL, shape=(3, 16, 16)) for key in cameras},
        },
        output_features={"action": types.PolicyFeature(type=types.FeatureType.ACTION, shape=(14,))},
    )
    cfg.validate_features()
    stats = {key: {"min": -torch.ones(14), "max": torch.ones(14)} for key in ("observation.state", "action")}
    pre, post = factory(cfg, dataset_stats=stats)
    batch = pre(
        {
            "observation.state": torch.zeros(14),
            "task": "synthetic environment smoke",
            **{key: torch.zeros(3, 16, 16) for key in cameras},
        }
    )
    assert batch["observation.state"].shape[-1] == 24
    raw = torch.zeros(1, 50, 24)
    raw[..., 6], raw[..., 13] = 0.25, -0.75
    result = post(raw)
    assert result.shape == (1, 50, 14)
    assert result.dtype == torch.float32
    assert torch.allclose(result[..., 6], torch.tensor(0.25), atol=1e-6)
    assert torch.allclose(result[..., 13], torch.tensor(-0.75), atol=1e-6)
    assert not torch.cuda.is_initialized()
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "prefix": sys.prefix,
        "versions": {
            name: version(name)
            for name in (
                "lerobot",
                "torch",
                "torchvision",
                "torchcodec",
                "transformers",
                "accelerate",
                "datasets",
                "av",
            )
        },
        "compiled_cuda": torch.version.cuda,
        "installed_packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
        "cuda_context_initialized": torch.cuda.is_initialized(),
        "imports": imports,
        "synthetic_processor_output_shape": list(result.shape),
        "status": "cpu_dependencies_and_synthetic_processors_passed_not_gpu_training_validated",
        "scope": "Synthetic 16px RGB and synthetic min/max; not actual YAM data, checkpoint or task validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
