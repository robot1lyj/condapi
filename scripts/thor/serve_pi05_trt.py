"""Serve an audited Pi0.5 YAM TensorRT engine on Thor's direct Ethernet link."""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
import numpy as np
import torch
from trt_policy import create_trt_policy

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi.training import config


class YAMInference:
    """The standard observation protocol; Thor owns only inference noise."""

    def __init__(self, policy):
        self.policy = policy
        self.noise_rng = np.random.default_rng()

    def infer(self, obs):
        if not isinstance(obs, dict):
            raise ValueError("Observation must be a dictionary")
        state = np.asarray(obs.get("observation.state"))
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("observation.state must be finite 14D")
        for view in ("top_rgb", "left_rgb", "right_rgb"):
            key = f"observation.images.{view}"
            image = np.asarray(obs.get(key))
            if image.ndim != 3 or image.dtype != np.uint8 or (image.shape[-1] != 3 and image.shape[0] != 3):
                raise ValueError(f"{key} must be CHW/HWC uint8 RGB")
        if not isinstance(obs.get("prompt"), str) or not obs["prompt"].strip():
            raise ValueError("A nonempty prompt is required")
        noise = self.noise_rng.standard_normal((50, 32)).astype(np.float32)
        result = self.policy.infer(obs, noise=noise)
        actions = np.asarray(result.get("actions"))
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise RuntimeError("Policy returned an invalid YAM action chunk")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--warmup-sample", type=Path, required=True)
    parser.add_argument("--host", default="192.168.250.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Thor CUDA is required")
    report = json.loads((args.engine / "engine_report.json").read_text())
    export = json.loads((Path(report["source_export"]) / "export_report.json").read_text())
    audit = json.loads((args.checkpoint / "conversion_audit.json").read_text())
    if (
        report["status"] != "built_not_inference_or_accuracy_validated"
        or export["status"] != "onnx_exported_engine_not_validated"
        or export["norm_stats_sha256"] != digest(args.norm)
        or export["conversion_audit_sha256"] != digest(args.checkpoint / "conversion_audit.json")
        or audit["output_precision"] != "float32"
        or not export["cache_time_modulation"]
        or export["text_bucket"] != 80
        or report["tf32"]
        or report["quantization"] is not None
    ):
        parser.error("Engine, FP32 checkpoint and training norm must match the accepted W route")
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, dtype="bfloat16"))
    policy = create_trt_policy(train, args.engine, normalize.deserialize_json(args.norm.read_text()))
    policy._model.enable_cuda_graph()  # noqa: SLF001
    serving = YAMInference(policy)
    sample = read_observation(args.warmup_sample)
    for _ in range(2):
        serving.infer(sample)
    metadata = {
        **policy.metadata,
        "config": "pi05_yam",
        "checkpoint_step": 100000,
        "checkpoint_weights_sha256": export["converted_weights_sha256"],
        "checkpoint_metadata_sha256": audit["checkpoint_metadata_sha256"],
        "norm_stats_sha256": export["norm_stats_sha256"],
        "engine_sha256": report["engine_sha256"],
        "backend": "tensorrt",
        "precision": "BF16 main compute; FP32 sensitive/time cache; nonquantized; TF32 off",
        "text_bucket": 80,
        "denoising_steps": 10,
        "output_action_dim": 14,
        "action_mode": "absolute target after checkpoint inverse transforms",
        "joint_unit": "radian (dataset contract; hardware calibration not claimed)",
        "gripper_unit": "continuous; nominal 0 closed, 1 open; outputs not clipped",
        "rtc_mode": "off",
    }
    logging.info("READY ws://%s:%d checkpoint=100000 backend=tensorrt", args.host, args.port)
    WebsocketPolicyServer(serving, host=args.host, port=args.port, rtc_mode="off", metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
