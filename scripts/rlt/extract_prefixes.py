# ruff: noqa: E402, PLC0415
"""GPU-server export of frozen final Pi image-prefix embeddings; no optimizer."""

import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "packages/parts-rl/src", ROOT / "packages/vla-platform/src", ROOT / "scripts/thor"):
    sys.path.insert(0, str(path))

from parts_rl.rlt_contract import PREFIX_SCHEMA
from parts_rl.rlt_data import load_observations
from parts_rl.rlt_data import member_path
from vla_platform import parts as wire

from adapters.parts.common import require_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="original FP32 PyTorch Pi asset directory")
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-host", action="append", required=True)
    parser.add_argument("--execute-on-server", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    observations = load_observations(args.observations)
    base_identity = {
        "checkpoint_weights_sha256": wire.sha256(args.checkpoint / "model.safetensors"),
        "norm_stats_sha256": wire.sha256(args.norm),
    }
    if args.check_only:
        print(json.dumps({"status": "inputs_checked_not_model_validated", "base_identity": base_identity}))
        return
    require_server({"allowed_training_hosts": args.server_host}, args.execute_on_server)
    if args.output.exists():
        parser.error("Prefix output must be a new directory")
    from benchmark_pi05 import read_observation
    import numpy as np
    from parts_rl.rlt_features import FrozenRltFeatures
    import torch

    from openpi.policies.policy_config import create_trained_policy
    from openpi.shared import normalize
    from openpi.training import config

    if not torch.cuda.is_available():
        parser.error("Server CUDA required")
    train = config.get_config("pi05_yam")
    rtc_path = args.checkpoint / "rtc_manifest.json"
    if rtc_path.is_file():
        rtc = json.loads(rtc_path.read_text())
        if (
            rtc.get("route") != "pi05_yam_trained_rtc"
            or rtc.get("model_weights_sha256") != base_identity["checkpoint_weights_sha256"]
            or rtc.get("norm_stats_sha256") != base_identity["norm_stats_sha256"]
        ):
            parser.error("Trained RTC asset binding mismatch")
        train = dataclasses.replace(
            train, model=dataclasses.replace(train.model, rtc_training_max_delay=rtc["max_delay_steps"])
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = create_trained_policy(
        train,
        args.checkpoint,
        norm_stats=normalize.deserialize_json(args.norm.read_text()),
        pytorch_device="cuda",
        pytorch_precision="float32",
        pytorch_compile=False,
        sample_kwargs={"num_steps": 5},
    )
    policy._model.requires_grad_(requires_grad=False).eval()  # noqa: SLF001
    args.output.mkdir(parents=True, exist_ok=False)
    members = []
    for source in observations["observations"]:
        relative = f"prefix_{len(members):08d}.npz"
        captured = []

        def sink(prefix, mask, captured=captured):
            captured.append((prefix[0].float().cpu().numpy(), mask[0].bool().cpu().numpy()))

        feature = FrozenRltFeatures(policy._model, prefix_sink=sink)  # noqa: SLF001
        with feature.capture():
            policy.infer(read_observation(member_path(args.observations.resolve().parent, source)))
        if len(captured) != 1:
            raise ValueError("Missing/ambiguous frozen final prefix")
        prefix, mask = captured[0]
        if not np.isfinite(prefix).all() or not mask.any():
            raise ValueError("Invalid frozen final prefix")
        np.savez_compressed(args.output / relative, prefix=prefix, mask=mask)
        members.append(
            {
                "path": relative,
                "sha256": wire.sha256(args.output / relative),
                "observation_key": source["observation_key"],
                "group_id": source["group_id"],
                "source_observation_sha256": source["sha256"],
            }
        )
    value = {
        "schema": "yam_rlt_prefixes_v1",
        "status": "READY",
        "prefix_schema": PREFIX_SCHEMA,
        "base_identity": base_identity,
        "mock": False,
        "split_role": "train",
        "train_groups": observations["train_groups"],
        "holdout_groups": observations["holdout_groups"],
        "observations_manifest_sha256": wire.sha256(args.observations),
        "members": members,
    }
    (args.output / "PREFIX_READY.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(args.output / "PREFIX_READY.json"), "observations": len(members)}))


if __name__ == "__main__":
    main()
