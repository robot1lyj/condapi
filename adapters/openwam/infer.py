"""Offline YAM action-chunk inference with self-contained native OpenWAM checkpoints."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.openwam.common import ROOT
from adapters.openwam.common import activate_source
from adapters.openwam.common import validate_data_config
from adapters.openwam.common import verify_source

sys.path.insert(0, str(ROOT / "packages/vla-platform/src"))
from vla_platform.contracts import validate_request
from vla_platform.contracts import validate_response
from vla_platform.project import read_toml
from vla_platform.project import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    activate_source()
    verify_source()
    contract = read_toml(args.contract)
    request = validate_request(json.loads(args.request.read_text()), contract)
    if args.output.exists():
        parser.error("Output already exists")
    import numpy as np  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    saved = OmegaConf.load(args.checkpoint / "config.yaml")
    plain = OmegaConf.to_container(saved, resolve=True)
    validate_data_config(plain["dataloader"], plain["model"]["architecture"])
    from openwam.deploy.engine import JointInferenceEngine  # noqa: PLC0415
    from openwam.deploy.model_loader import load_from_checkpoint_dir  # noqa: PLC0415
    from openwam.deploy.server import merge_deploy_cfg  # noqa: PLC0415

    from adapters.openwam.data import compose_image  # noqa: PLC0415

    # Saved fps travels with the checkpoint; never guess Pi's action period.
    fps = float(saved.dataloader.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Checkpoint must contain audited positive dataset fps")
    images = {}
    for key, reference in request["images"].items():
        path = Path(reference)
        with Image.open(path if path.is_absolute() else args.request.parent / path) as image:
            images[key] = image.copy()
    canvas = compose_image(images, plain["dataloader"])
    cfg, architecture = load_from_checkpoint_dir(str(args.checkpoint), device=args.device)
    deploy = OmegaConf.create(
        {
            "inference": {"denoise_steps": 10, "denoise_mode": "sync"},
            "optimization": {
                "decode_video": False,
                "dit_cache": {"enabled": False},
                "compile": {"enabled": False},
                "prompt_embed_cache": {"enabled": False},
            },
        }
    )
    engine = JointInferenceEngine(merge_deploy_cfg(cfg, deploy), architecture=architecture)
    started = time.perf_counter()
    actions = np.asarray(
        engine.generate(
            {
                "first_frame_image": [canvas],
                "prompt": request["prompt"],
                "proprio": np.asarray(request["state"], dtype=np.float32)[None],
            }
        )["actions"]
    )
    elapsed = (time.perf_counter() - started) * 1000
    if actions.shape != (saved.dataloader.num_frames - 1, 14):
        raise ValueError(f"Unexpected YAM action chunk shape: {actions.shape}")
    response = {
        "schema_version": 1,
        "request_id": request["request_id"],
        "session_id": request["session_id"],
        "contract_id": contract["id"],
        "model_version": args.model_version,
        "actions": actions.tolist(),
        "action_dt_s": 1 / fps,
        "latency_ms": elapsed,
        "output_semantics": "absolute",
    }
    validate_response(response, contract)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "response.json", response)
    write_json(
        args.output / "scope.json",
        {
            "status": "offline_reference_only",
            "reset": "new_process_per_request",
            "acceptance": "Thor and cross-IPC acceptance remain required",
        },
    )


if __name__ == "__main__":
    main()
