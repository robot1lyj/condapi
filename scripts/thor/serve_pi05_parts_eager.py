"""Opt-in eager FP32 Pi/PARTS service; GPU numerical/latency acceptance is external.

Uses the existing Pi loader and trained RTC adapter. Never converts checkpoints
or starts an optimizer. The production TRT entry remains separately available.
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
from parts_policy import load_eager_extension
from parts_policy import wire
from rtc_policy import RtcEagerAdapter
from rtc_policy import TrainedRtcInference
import torch

from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--parts-manifest", type=Path, required=True)
    parser.add_argument("--warmup-sample", type=Path, required=True)
    parser.add_argument("--warmup-rtc", type=Path)
    parser.add_argument("--rtc-mode", choices=("off", "trained"), required=True)
    parser.add_argument("--num-steps", type=int, choices=(5, 6, 7, 8, 10), required=True)
    parser.add_argument("--max-joint-step-rad", type=float, required=True)
    parser.add_argument("--host", default="192.168.250.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Thor CUDA required")
    weights = args.checkpoint / "model.safetensors"
    weights_sha, norm_sha = digest(weights), digest(args.norm)
    train = config.get_config("pi05_yam")
    trained = args.rtc_mode == "trained"
    max_delay = None
    if trained:
        rtc_manifest = json.loads((args.checkpoint / "rtc_manifest.json").read_text())
        if (
            rtc_manifest.get("route") != "pi05_yam_trained_rtc"
            or rtc_manifest.get("model_weights_sha256") != weights_sha
            or rtc_manifest.get("norm_stats_sha256") != norm_sha
            or args.warmup_rtc is None
        ):
            parser.error("Trained RTC checkpoint/norm/warmup mismatch")
        max_delay = rtc_manifest["max_delay_steps"]
        train = dataclasses.replace(train, model=dataclasses.replace(train.model, rtc_training_max_delay=max_delay))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stats = normalize.deserialize_json(args.norm.read_text())
    policy = create_trained_policy(
        train,
        args.checkpoint,
        norm_stats=stats,
        pytorch_device="cuda",
        pytorch_precision="float32",
        pytorch_compile=False,
        sample_kwargs={"num_steps": args.num_steps},
    )
    metadata = {
        **policy.metadata,
        "config": "pi05_yam",
        "backend": "eager_parts_candidate",
        "status": "not_robot_or_latency_validated",
        "rtc_mode": args.rtc_mode,
        "checkpoint_weights_sha256": weights_sha,
        "norm_stats_sha256": norm_sha,
        "action_horizon": 50,
        "action_dim": 14,
        "action_dt_s": 1 / 30,
        "rtc_max_delay_steps": max_delay,
        "compute_dtype": "float32",
    }
    extension = load_eager_extension(args.parts_manifest, policy, metadata)
    serving = policy
    if trained:
        use_quantiles = train.data.create(train.assets_dirs, train.model).use_quantile_norm
        serving = TrainedRtcInference(
            policy,
            stats,
            max_delay=max_delay,
            use_quantiles=use_quantiles,
            num_steps=args.num_steps,
            sampler=RtcEagerAdapter(
                policy._model,  # noqa: SLF001
                max_delay=max_delay,
            ),
            max_joint_step_rad=args.max_joint_step_rad,
        )
    sample = read_observation(args.warmup_sample)
    rtc = json.loads(args.warmup_rtc.read_text()) if trained else None
    with extension.features.capture() as features:
        result = serving.infer_rtc(sample, rtc) if trained else serving.infer(sample)

    wire.numeric_array(result.get("actions"), (50, 14), "warmup actions")
    if features.z is None or features.z.shape != (extension.feature_dim,):
        parser.error("Warmup feature dimension differs from pinned manifest")
    logging.info("Eager PARTS candidate listening; numerical/latency/task acceptance remains required")
    WebsocketPolicyServer(
        serving, host=args.host, port=args.port, rtc_mode=args.rtc_mode, metadata=metadata, parts_extension=extension
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
