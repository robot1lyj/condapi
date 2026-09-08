"""Offline Pi reference adapter. GPU deployment/IPC are separate acceptance steps."""

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/vla-platform/src"))
sys.path.insert(0, str(ROOT / "src"))

from vla_platform.contracts import validate_request  # noqa: E402
from vla_platform.contracts import validate_response  # noqa: E402
from vla_platform.project import read_toml  # noqa: E402
from vla_platform.project import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", choices=("pi05_yam", "pi05_yam_lora"), required=True)
    parser.add_argument("--backend", choices=("jax", "pytorch"), required=True)
    parser.add_argument("--compute", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--action-dt", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_toml(args.contract)
    request = validate_request(json.loads(args.request.read_text()), contract)
    if not args.checkpoint.is_dir():
        parser.error("Checkpoint must be an existing local directory")
    is_torch = (args.checkpoint / "model.safetensors").is_file()
    if is_torch != (args.backend == "pytorch"):
        parser.error("Checkpoint format differs from explicitly selected backend")
    if is_torch and args.config.endswith("lora"):
        parser.error("Unmerged LoRA is not supported by this PyTorch adapter")
    if args.action_dt <= 0:
        parser.error("Action period must be positive, from dataset metadata")
    if args.output.exists():
        parser.error("Output already exists")

    # Heavy dependencies are confined to the selected Conda worker.
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from openpi.policies.policy_config import create_trained_policy  # noqa: PLC0415
    from openpi.training.config import get_config  # noqa: PLC0415

    config = get_config(args.config)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, dtype=args.compute))
    policy = create_trained_policy(
        config,
        args.checkpoint,
        jax_param_dtype="checkpoint",
        pytorch_precision=args.compute,
        pytorch_compile=False,
        sample_kwargs={"num_steps": 10},
    )
    observation = {"observation.state": np.asarray(request["state"], dtype=np.float32), "prompt": request["prompt"]}
    for key, reference in request["images"].items():
        image_path = Path(reference)
        if not image_path.is_absolute():
            image_path = args.request.parent / image_path
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB image, got {image.mode}: {image_path}")
            observation[key] = np.asarray(image).copy()
    # CPU materialization completes the call before measuring end-to-end latency.
    started = time.perf_counter()
    actions = np.asarray(policy.infer(observation)["actions"]).tolist()
    elapsed = (time.perf_counter() - started) * 1000
    response = {
        "schema_version": 1,
        "request_id": request["request_id"],
        "session_id": request["session_id"],
        "contract_id": contract["id"],
        "model_version": args.model_version,
        "actions": actions,
        "action_dt_s": args.action_dt,
        "latency_ms": elapsed,
        "output_semantics": "absolute",
    }
    validate_response(response, contract)
    if len(actions) != 50:
        raise ValueError("Pi adapter requires H50")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "response.json", response)
    write_json(
        args.output / "scope.json",
        {
            "status": "offline_reference_only",
            "backend": args.backend,
            "compute": args.compute,
            "weight_load": "checkpoint_preserved" if not is_torch else "explicit_compute",
            "timing": "first_call_including_compilation_not_steady_state_benchmark",
            "reset": "new_process_per_request",
            "production_accepted": False,
        },
    )


if __name__ == "__main__":
    main()
