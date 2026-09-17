"""Serve an independently validated trained-RTC TensorRT engine on Thor."""

import argparse
import dataclasses
import json
import logging
import math
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
from rtc_policy import TrainedRtcInference
from rtc_trt_policy import RtcTensorRTAdapter
from rtc_trt_policy import create_rtc_transform_policy
import torch

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi.training import config


def check_validated_tf32_7step(report, export, manifest, validation):
    """Limit production promotion to the independently replayed RTC candidate."""
    cases = validation.get("cases", [])
    if (
        report.get("status") != "built_experiment_not_accuracy_validated"
        or report.get("exit_code") != 0
        or report.get("precision_candidate_kind") != "tf32"
        or report.get("tf32") is not True
        or report.get("quantization") is not None
        or report.get("strongly_typed") is not True
        or export.get("compute_dtype") != "float32"
        or export.get("contract", {}).get("steps") != 7
        or validation.get("status") != "compared_not_robot_task_validated"
        or validation.get("engine_sha256") != report.get("engine_sha256")
        or validation.get("engine_tf32") is not True
        or validation.get("num_steps") != 7
        or validation.get("checkpoint_weights_sha256") != manifest.get("model_weights_sha256")
        or validation.get("prefix_use_quantiles") is not True
        or validation.get("checkpoint_metadata_sha256") != manifest.get("checkpoint_metadata_sha256")
        or validation.get("norm_stats_sha256") != manifest.get("norm_stats_sha256")
        or validation.get("cases_sha256") != export.get("cases_sha256")
        or len(cases) != 9
        or {case.get("delay_steps") for case in cases} != {0, 1, 10}
        or not all(case.get("finite") is True and case.get("prefix_exact") is True for case in cases)
        or not math.isfinite(validation.get("physical_max_abs", float("nan")))
        or validation["physical_max_abs"] > 0.005
        or not math.isfinite(validation.get("physical_p99_abs", float("nan")))
        or validation["physical_p99_abs"] > 0.001
    ):
        raise ValueError("The seven-step TF32 engine lacks matching numerical validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--warmup-sample", type=Path, required=True)
    parser.add_argument("--warmup-rtc", type=Path, required=True)
    parser.add_argument("--host", default="192.168.250.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--allow-validated-tf32-7step", action="store_true")
    parser.add_argument("--validation", type=Path, help="new quantile-prefix validation receipt")
    parser.add_argument("--jax-reference", type=Path, help="corrected original-JAX reference directory")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Thor CUDA required")
    report = json.loads((args.engine / "engine_report.json").read_text())
    export_path = Path(report["source_export"]) / "export_report.json"
    export = json.loads(export_path.read_text())
    manifest_path = args.checkpoint / "rtc_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    experimental = args.allow_validated_tf32_7step
    if (
        (not experimental and report["status"] != "built_not_inference_or_accuracy_validated")
        or report["exit_code"] != 0
        or export["status"] != "onnx_exported_engine_not_validated"
        or export["rtc_mode"] != "trained"
        or export["norm_stats_sha256"] != digest(args.norm)
        or export["rtc_manifest_sha256"] != digest(manifest_path)
        or manifest["route"] != "pi05_yam_trained_rtc"
        or not all(row["numeric_gate_1e-4"] for row in export["jax_comparisons"])
        or (report["tf32"] and not experimental)
        or report["quantization"] is not None
    ):
        parser.error("RTC engine/checkpoint/norm/JAX gate mismatch")
    if experimental:
        if args.validation is None or args.jax_reference is None:
            parser.error("Corrected quantile-prefix validation and JAX reference are required")
        validation = json.loads(args.validation.read_text())
        check_validated_tf32_7step(report, export, manifest, validation)
        reference_path = args.jax_reference / "reference_manifest.json"
        reference = json.loads(reference_path.read_text())
        if (
            reference.get("prefix_use_quantiles") is not True
            or reference.get("checkpoint_metadata_sha256") != manifest["checkpoint_metadata_sha256"]
            or reference.get("norm_stats_sha256") != digest(args.norm)
            or reference.get("cases_sha256") != validation["cases_sha256"]
            or validation["jax_reference_manifest_sha256"] != digest(reference_path)
        ):
            parser.error("RTC corrected JAX reference is not bound to the validation receipt")
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(
        train.model, dtype=export["compute_dtype"], rtc_training_max_delay=manifest["max_delay_steps"],
    ))
    stats = normalize.deserialize_json(args.norm.read_text())
    use_quantiles = train.data.create(train.assets_dirs, train.model).use_quantile_norm
    if not use_quantiles:
        parser.error("pi05_yam must use quantile normalization for RTC")
    policy = create_rtc_transform_policy(train, stats)
    backend = RtcTensorRTAdapter(
        args.engine, max_delay=manifest["max_delay_steps"], allow_experimental=experimental
    )
    backend.enable_cuda_graph()
    serving = TrainedRtcInference(
        policy, stats, max_delay=manifest["max_delay_steps"], use_quantiles=use_quantiles,
        text_bucket=export["text_bucket"], num_steps=export["contract"]["steps"], sampler=backend,
        max_joint_step_rad=0.2,
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
        "rtc_prefix_norm": "quantile" if use_quantiles else "mean_std",
        "rtc_joint_step_guard_rad_per_tick": 0.2,
        "rtc_semantics": "clean aligned committed prefix at flow time zero; postfix denoised; no guidance/fallback",
        "rtc_max_delay_steps": manifest["max_delay_steps"],
        "checkpoint_weights_sha256": manifest["model_weights_sha256"],
        "norm_stats_sha256": manifest["norm_stats_sha256"],
        "engine_sha256": report["engine_sha256"],
        "compute_dtype": export["compute_dtype"],
        "precision_mode": (
            "fp32_weights_tf32_compute" if experimental else f"{export['compute_dtype']}_no_tf32"
        ),
        "action_horizon": 50,
        "denoising_steps": export["contract"]["steps"],
        "action_dt_s": 1 / 30,
        "action_dim": 14,
        "action_mode": "absolute physical target after checkpoint inverse transforms",
        "joint_unit": "radian (dataset contract; hardware calibration not claimed)",
        "gripper_unit": "continuous; nominal 0 closed, 1 open; outputs not clipped",
        "target_tick_semantics": "30Hz policy tick; observation_policy_tick == target_start_tick == committed_start_tick",
        "action_0_relative_to_observation_policy_tick": 0,
        "camera_exposure_to_policy_tick_offset_s": None,
    }
    logging.info("READY ws://%s:%d trained-RTC", args.host, args.port)
    WebsocketPolicyServer(serving, host=args.host, port=args.port, rtc_mode="trained", metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
