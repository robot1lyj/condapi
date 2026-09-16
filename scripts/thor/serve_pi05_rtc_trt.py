"""Serve an independently validated trained-RTC TensorRT engine on Thor."""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
from rtc_policy import TrainedRtcInference
from rtc_trt_policy import create_rtc_transform_policy
from rtc_trt_policy import RtcTensorRTAdapter
import torch

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--warmup-sample", type=Path, required=True)
    parser.add_argument("--warmup-rtc", type=Path, required=True)
    parser.add_argument("--host", default="192.168.250.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Thor CUDA required")
    report = json.loads((args.engine / "engine_report.json").read_text())
    export_path = Path(report["source_export"]) / "export_report.json"
    export = json.loads(export_path.read_text())
    manifest_path = args.checkpoint / "rtc_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        report["status"] != "built_not_inference_or_accuracy_validated"
        or report["exit_code"] != 0
        or export["status"] != "onnx_exported_engine_not_validated"
        or export["rtc_mode"] != "trained"
        or export["norm_stats_sha256"] != digest(args.norm)
        or export["rtc_manifest_sha256"] != digest(manifest_path)
        or manifest["route"] != "pi05_yam_trained_rtc"
        or not all(row["numeric_gate_1e-4"] for row in export["jax_comparisons"])
        or report["tf32"]
        or report["quantization"] is not None
    ):
        parser.error("RTC engine/checkpoint/norm/JAX gate mismatch")
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(
        train.model, dtype=export["compute_dtype"], rtc_training_max_delay=manifest["max_delay_steps"],
    ))
    stats = normalize.deserialize_json(args.norm.read_text())
    policy = create_rtc_transform_policy(train, stats)
    backend = RtcTensorRTAdapter(args.engine, max_delay=manifest["max_delay_steps"])
    backend.enable_cuda_graph()
    serving = TrainedRtcInference(
        policy, stats, max_delay=manifest["max_delay_steps"],
        text_bucket=export["text_bucket"], sampler=backend,
    )
    warmup = read_observation(args.warmup_sample)
    warmup_rtc = json.loads(args.warmup_rtc.read_text())
    for _ in range(2):
        serving.infer_rtc(warmup, warmup_rtc)
    metadata = {
        **policy.metadata,
        "config": "pi05_yam",
        "backend": "tensorrt_cuda_graph",
        "rtc_mode": "trained",
        "rtc_semantics": "clean aligned committed prefix at flow time zero; postfix denoised; no guidance/fallback",
        "rtc_max_delay_steps": manifest["max_delay_steps"],
        "checkpoint_weights_sha256": manifest["model_weights_sha256"],
        "norm_stats_sha256": manifest["norm_stats_sha256"],
        "engine_sha256": report["engine_sha256"],
        "compute_dtype": export["compute_dtype"],
        "action_horizon": 50,
        "action_dim": 14,
        "action_mode": "absolute physical target after checkpoint inverse transforms",
        "joint_unit": "radian (dataset contract; hardware calibration not claimed)",
        "gripper_unit": "continuous; nominal 0 closed, 1 open; outputs not clipped",
        "target_tick_semantics": "controller tick; committed_start_tick must equal target_start_tick",
        "first_step_observation_offset_s": None,
    }
    logging.info("READY ws://%s:%d trained-RTC", args.host, args.port)
    WebsocketPolicyServer(serving, host=args.host, port=args.port, rtc_mode="trained", metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
