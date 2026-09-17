"""Compare the trained-RTC TensorRT engine with original JAX on real YAM cases."""

import argparse
import dataclasses
import datetime
import json
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
import numpy as np
from rtc_policy import TrainedRtcInference
from rtc_trt_policy import create_rtc_transform_policy
from rtc_trt_policy import RtcTensorRTAdapter
import torch

from openpi.shared import normalize
from openpi.training import config


def checked_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("RTC case path escaped fixture directory")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--jax-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available():
        parser.error("Use a new output path on Thor CUDA")
    report = json.loads((args.engine / "engine_report.json").read_text())
    export = json.loads((Path(report["source_export"]) / "export_report.json").read_text())
    manifest = json.loads((args.checkpoint / "rtc_manifest.json").read_text())
    norm_path = args.checkpoint / "assets" / "yam" / "norm_stats.json"
    case_set = json.loads(args.cases.read_text())
    references = json.loads((args.jax_reference / "reference_manifest.json").read_text())
    if (
        report["status"] != "built_not_inference_or_accuracy_validated"
        or export["status"] != "onnx_exported_engine_not_validated"
        or export["rtc_manifest_sha256"] != digest(args.checkpoint / "rtc_manifest.json")
        or export["norm_stats_sha256"] != digest(norm_path)
        or export["cases_sha256"] != digest(args.cases)
        or export["jax_reference_manifest_sha256"] != digest(args.jax_reference / "reference_manifest.json")
        or case_set["source_kind"] != "real_yam_recording"
        or len(references["cases"]) != len(case_set["cases"])
    ):
        raise ValueError("RTC engine/checkpoint/cases/JAX reference mismatch")
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(
        train.model, dtype=export["compute_dtype"], rtc_training_max_delay=manifest["max_delay_steps"],
    ))
    stats = normalize.deserialize_json(norm_path.read_text())
    policy = create_rtc_transform_policy(train, stats)
    backend = RtcTensorRTAdapter(args.engine, max_delay=manifest["max_delay_steps"])
    backend.enable_cuda_graph()
    serving = TrainedRtcInference(policy, stats, max_delay=manifest["max_delay_steps"], sampler=backend)
    noise = np.random.default_rng(0).standard_normal((1, 50, 32)).astype(np.float32)
    rows = []
    for index, case in enumerate(case_set["cases"]):
        ref_row = references["cases"][index]
        if ref_row["sample"] != case["sample"] or ref_row["delay_steps"] != case["delay_steps"]:
            raise ValueError("RTC reference case order mismatch")
        ref_path = checked_path(args.jax_reference, ref_row["reference"])
        if digest(ref_path) != ref_row["sha256"]:
            raise ValueError("RTC JAX reference changed")
        with np.load(ref_path, allow_pickle=False) as reference:
            expected = reference["physical"].copy()
        obs = read_observation(checked_path(args.cases.parent, case["sample"]))
        committed = np.load(checked_path(args.cases.parent, case["committed_actions"]), allow_pickle=False)
        rtc = {
            "delay_steps": case["delay_steps"],
            "observation_policy_tick": case["observation_policy_tick"],
            "target_start_tick": case["target_start_tick"],
            "committed_start_tick": case["committed_start_tick"],
            "committed_actions": committed,
        }
        result = serving.infer_rtc(obs, rtc, noise=noise)
        actual = np.asarray(result["actions"])
        delay = case["delay_steps"]
        if (
            actual.shape != (50, 14)
            or not np.isfinite(actual).all()
            or not np.array_equal(actual[:delay], committed.astype(np.float32))
        ):
            raise RuntimeError(f"Invalid RTC TensorRT output: {case['sample']}")
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        rows.append({
            "sample": case["sample"],
            "delay_steps": delay,
            "physical_max_abs": float(error.max()),
            "physical_mean_abs": float(error.mean()),
            "physical_p99_abs": float(np.percentile(error, 99)),
            "server_infer_ms": float(result["policy_timing"]["infer_ms"]),
            "finite": True,
            "prefix_exact": True,
        })
    receipt = {
        "status": "compared_not_robot_task_validated",
        "observed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "engine_sha256": report["engine_sha256"],
        "checkpoint_weights_sha256": manifest["model_weights_sha256"],
        "jax_reference_manifest_sha256": digest(args.jax_reference / "reference_manifest.json"),
        "cases": rows,
        "physical_max_abs": max(row["physical_max_abs"] for row in rows),
        "physical_mean_abs": float(np.mean([row["physical_mean_abs"] for row in rows])),
        "server_infer_p50_ms": float(np.percentile([row["server_infer_ms"] for row in rows], 50)),
        "server_infer_p95_ms": float(np.percentile([row["server_infer_ms"] for row in rows], 95)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
