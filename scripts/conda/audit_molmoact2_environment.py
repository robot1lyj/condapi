"""Offline CPU imports, native configuration and synthetic 14D normalization only."""

import argparse
from datetime import UTC
from datetime import datetime
import importlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import sys


def audit():
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", WANDB_MODE="disabled", OMP_NUM_THREADS="1")
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    modules = [
        "torchvision",
        "torchcodec.decoders",
        "av",
        "peft",
        "scipy",
        "transformers",
        "accelerate",
        "lerobot.scripts.lerobot_train",
        "lerobot.policies.molmoact2.modeling_molmoact2",
    ]
    for name in modules:
        importlib.import_module(name)
    config_module = importlib.import_module("lerobot.policies.molmoact2.configuration_molmoact2")
    processor = importlib.import_module("lerobot.policies.molmoact2.processor_molmoact2")
    types = importlib.import_module("lerobot.configs.types")
    pipeline = importlib.import_module("lerobot.processor")
    converters = importlib.import_module("lerobot.processor.converters")
    cameras = [f"observation.images.{camera}_rgb" for camera in ("top", "left", "right")]
    names = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(7)]
    names[6], names[13] = "left_gripper", "right_gripper"
    cfg = config_module.MolmoAct2Config(
        device="cpu",
        dtype="float32",
        compile_model=False,
        enable_inference_cuda_graph=False,
        train_mode_vlm="lora",
        image_keys=cameras,
        input_features={
            "observation.state": types.PolicyFeature(type=types.FeatureType.STATE, shape=(14,)),
            **{key: types.PolicyFeature(type=types.FeatureType.VISUAL, shape=(3, 16, 16)) for key in cameras},
        },
        output_features={"action": types.PolicyFeature(type=types.FeatureType.ACTION, shape=(14,))},
        dataset_feature_names={"action": names, "observation.state": names},
    )
    cfg.validate_features()
    stats = {
        key: {"min": -torch.ones(14), "max": torch.ones(14), "q01": -torch.ones(14), "q99": torch.ones(14)}
        for key in ("action", "observation.state")
    }
    masked = processor._add_gripper_masks_to_stats(  # noqa: SLF001 -- audit the pinned native mask helper
        stats,
        None,
        normalize_gripper=False,
        dataset_feature_names=cfg.dataset_feature_names,
    )
    expected_mask = [True] * 14
    expected_mask[6] = expected_mask[13] = False
    assert masked["action"]["mask"] == expected_mask
    post = pipeline.PolicyProcessorPipeline(
        steps=[
            processor.MolmoAct2MaskedUnnormalizerProcessorStep(
                features=cfg.output_features,
                norm_map=cfg.normalization_mapping,
                stats=masked,
            )
        ],
        to_transition=converters.policy_action_to_transition,
        to_output=converters.transition_to_policy_action,
    )
    action = torch.zeros(1, cfg.chunk_size, 14)
    action[..., 6], action[..., 13] = 0.25, -0.75
    result = post(action)
    assert torch.allclose(result, action, atol=1e-6)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = config_module.MolmoAct2AdamW([parameter], lr=1e-3, group_grad_clip_norm=1.0)
    parameter.square().sum().backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert parameter.item() < 1.0
    assert not torch.cuda.is_initialized()
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "prefix": sys.prefix,
        "python": sys.version,
        "compiled_cuda": torch.version.cuda,
        "installed_packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
        "imports": modules,
        "action_shape": list(result.shape),
        "gripper_normalization_mask": expected_mask,
        "synthetic_fp32_optimizer_step": "passed",
        "cuda_context_initialized": False,
        "status": "cpu_dependencies_config_normalization_passed_not_model_validated",
        "scope": "No checkpoint/tokenizer/full processor/GPU or real YAM data; synthetic stats and one scalar optimizer only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "installed_packages"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
