"""Compare the trained-RTC TensorRT engine with original JAX on real YAM cases."""

import argparse
import dataclasses
import datetime
import json
from pathlib import Path
import time

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
    parser.add_argument("--allow-experimental", action="store_true")
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
        report["status"] != (
            "built_experiment_not_accuracy_validated" if args.allow_experimental
            else "built_not_inference_or_accuracy_validated"
        )
        or export["status"] != "onnx_exported_engine_not_validated"
        or export["rtc_manifest_sha256"] != digest(args.checkpoint / "rtc_manifest.json")
        or export["norm_stats_sha256"] != digest(norm_path)
        or export["cases_sha256"] != digest(args.cases)
        or export["jax_reference_manifest_sha256"] != digest(args.jax_reference / "reference_manifest.json")
        or case_set["source_kind"] != "real_yam_recording"
        or len(references["cases"]) != len(case_set["cases"])
        or references.get("num_steps", 10) != export["contract"]["steps"]
    ):
        raise ValueError("RTC engine/checkpoint/cases/JAX reference mismatch")
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(
        train.model, dtype=export["compute_dtype"], rtc_training_max_delay=manifest["max_delay_steps"],
    ))
    stats = normalize.deserialize_json(norm_path.read_text())
    policy = create_rtc_transform_policy(train, stats)
    backend = RtcTensorRTAdapter(
        args.engine, max_delay=manifest["max_delay_steps"], allow_experimental=args.allow_experimental
    )
    backend.enable_cuda_graph()
    serving = TrainedRtcInference(
        policy, stats, max_delay=manifest["max_delay_steps"],
        num_steps=export["contract"]["steps"], sampler=backend,
    )
    noise = np.random.default_rng(0).standard_normal((1, 50, 32)).astype(np.float32)
    rows = []
    requests = []
    errors = []
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
        started = time.perf_counter()
        result = serving.infer_rtc(obs, rtc, noise=noise)
        total_ms = (time.perf_counter() - started) * 1000
        actual = np.asarray(result["actions"])
        delay = case["delay_steps"]
        if (
            actual.shape != (50, 14)
            or not np.isfinite(actual).all()
            or not np.array_equal(actual[:delay], committed.astype(np.float32))
        ):
            raise RuntimeError(f"Invalid RTC TensorRT output: {case['sample']}")
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        errors.append(error)
        rows.append({
            "sample": case["sample"],
            "delay_steps": delay,
            "physical_max_abs": float(error.max()),
            "physical_mean_abs": float(error.mean()),
            "physical_p99_abs": float(np.percentile(error, 99)),
            "server_infer_ms": float(result["policy_timing"]["infer_ms"]),
            "server_total_ms": total_ms,
            "finite": True,
            "prefix_exact": True,
        })
        requests.append((obs, rtc))
    # First-pass checks above include lazy CUDA graph capture. Repeat real
    # observations after warmup to report the latency a persistent service sees.
    steady_infer_ms = []
    steady_total_ms = []
    for _ in range(3):
        for obs, rtc in requests:
            started = time.perf_counter()
            result = serving.infer_rtc(obs, rtc, noise=noise)
            steady_total_ms.append((time.perf_counter() - started) * 1000)
            steady_infer_ms.append(float(result["policy_timing"]["infer_ms"]))
    all_errors = np.stack(errors)
    worst_case, worst_step, worst_dim = np.unravel_index(int(all_errors.argmax()), all_errors.shape)
    receipt = {
        "status": "compared_not_robot_task_validated",
        "observed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "engine_sha256": report["engine_sha256"],
        "engine_tf32": report["tf32"],
        "num_steps": export["contract"]["steps"],
        "checkpoint_weights_sha256": manifest["model_weights_sha256"],
        "jax_reference_manifest_sha256": digest(args.jax_reference / "reference_manifest.json"),
        "cases": rows,
        "physical_max_abs": max(row["physical_max_abs"] for row in rows),
        "physical_mean_abs": float(np.mean([row["physical_mean_abs"] for row in rows])),
        "physical_p95_abs": float(np.percentile(all_errors, 95)),
        "physical_p99_abs": float(np.percentile(all_errors, 99)),
        "worst_element": {"case_index": int(worst_case), "action_step": int(worst_step), "dimension": int(worst_dim)},
        "per_dimension": [
            {
                "dimension": index,
                "kind": "gripper" if index in (6, 13) else "joint_rad",
                "max_abs": float(all_errors[:, :, index].max()),
                "mean_abs": float(all_errors[:, :, index].mean()),
                "p95_abs": float(np.percentile(all_errors[:, :, index], 95)),
            }
            for index in range(14)
        ],
        "server_infer_p50_ms": float(np.percentile([row["server_infer_ms"] for row in rows], 50)),
        "server_infer_p95_ms": float(np.percentile([row["server_infer_ms"] for row in rows], 95)),
        "steady_server_infer_p50_ms": float(np.percentile(steady_infer_ms, 50)),
        "steady_server_infer_p95_ms": float(np.percentile(steady_infer_ms, 95)),
        "steady_server_total_p50_ms": float(np.percentile(steady_total_ms, 50)),
        "steady_server_total_p95_ms": float(np.percentile(steady_total_ms, 95)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
