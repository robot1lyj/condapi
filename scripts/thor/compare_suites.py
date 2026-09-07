"""Compare immutable completed suites; never treat measured drift as approval."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def error_metrics(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Finite, identical action shapes required")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(absolute.max()),
        "worst_index": [int(i) for i in np.unravel_index(np.argmax(absolute), absolute.shape)],
        "per_dimension_max_abs": absolute.reshape(-1, absolute.shape[-1]).max(axis=0).tolist(),
        "per_dimension_mae": absolute.reshape(-1, absolute.shape[-1]).mean(axis=0).tolist(),
    }


def read_run(path):
    path = Path(path)
    record = json.loads((path / "result.json").read_text())
    if record["status"] != "measured_not_accuracy_approved":
        raise ValueError("Only completed measured suites can be compared")
    if digest(path / "noise.npy") != record["noise_sha256"]:
        raise ValueError("Noise artifact hash mismatch")
    arrays = {}
    for key in ("actions", "normalized_actions"):
        file = path / f"{key}.npy"
        if digest(file) != record[f"{key}_sha256"]:
            raise ValueError("Action artifact hash mismatch")
        array = np.load(file, allow_pickle=False)
        dimensions = 14 if key == "actions" else 32
        expected = (len(record["measurements"]), record["repeats"], 50, dimensions)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError("Unexpected action shape or values")
        arrays[key] = array
    return record, arrays


def compare(reference_path, candidate_path, *, cross_backend=False):
    ref, ref_arrays = read_run(reference_path)
    cand, cand_arrays = read_run(candidate_path)
    for key in (
        "suite_sha256",
        "norm_stats_sha256",
        "checkpoint_metadata_sha256",
        "noise_sha256",
        "seed",
        "steps",
        "horizon",
        "repeats",
        "config",
    ):
        if ref[key] != cand[key]:
            raise ValueError(f"Confounded comparison: {key}")
    runtime_differences = {}
    for key in ("versions", "matmul_precision", "backend"):
        reference = ref.get(key, "jax")
        candidate = cand.get(key, "jax")
        if reference != candidate:
            if not cross_backend:
                raise ValueError(f"Confounded comparison: {key}")
            runtime_differences[key] = {"reference": reference, "candidate": candidate}
    if [m["sample_sha256"] for m in ref["measurements"]] != [m["sample_sha256"] for m in cand["measurements"]]:
        raise ValueError("Sample ordering mismatch")
    # First measured repeat per observation for precision error; repeated runs
    # only estimate timing/jitter, not an artificially enlarged dataset size.
    physical = error_metrics(ref_arrays["actions"][:, 0], cand_arrays["actions"][:, 0])
    normalized = error_metrics(
        ref_arrays["normalized_actions"][:, 0, :, :14], cand_arrays["normalized_actions"][:, 0, :, :14]
    )
    per_sample = []
    for index, sample in enumerate(ref["measurements"]):
        per_sample.append(
            {
                "sample": sample["sample"],
                "physical": error_metrics(ref_arrays["actions"][index, 0], cand_arrays["actions"][index, 0]),
                "normalized": error_metrics(
                    ref_arrays["normalized_actions"][index, 0, :, :14],
                    cand_arrays["normalized_actions"][index, 0, :, :14],
                ),
            }
        )
    return {
        "reference": ref["run_id"],
        "candidate": cand["run_id"],
        "physical_dataset_units": physical,
        "normalized_active_14d": normalized,
        "normalized_internal_32d": error_metrics(
            ref_arrays["normalized_actions"][:, 0], cand_arrays["normalized_actions"][:, 0]
        ),
        "per_sample": per_sample,
        "speedup_p50": ref["p50_ms"] / cand["p50_ms"],
        "acceptance": "not_approved_no_task_level_tolerance",
        "comparison_scope": "cross_backend_including_runtime_changes" if cross_backend else "same_runtime_precision",
        "runtime_differences": runtime_differences,
    }


def telemetry_metrics(path):
    text = Path(path).read_text()
    ram = [int(value) for value in re.findall(r"RAM (\d+)/", text)]
    gpu_temp = [float(value) for value in re.findall(r"gpu@([\d.]+)C", text)]
    clocks = [int(value) for group in re.findall(r"GR3D_FREQ @\[([\d,]+)\]", text) for value in group.split(",")]
    watts = [int(value) / 1000 for value in re.findall(r"\bVIN (\d+)mW", text)]
    if not all((ram, gpu_temp, clocks, watts)):
        raise ValueError("Missing required Thor telemetry")
    return {
        "samples": len(ram),
        "system_ram_peak_mib": max(ram),
        "gpu_temperature_max_c": max(gpu_temp),
        "gpu_clock_min_mhz": min(clocks),
        "gpu_clock_max_mhz": max(clocks),
        "input_power_max_w": max(watts),
        "scope": "whole_inference_process_including_loading_and_warmup; RAM includes OS/desktop",
        "telemetry_sha256": digest(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-backend", action="store_true")
    args = parser.parse_args()
    result = {
        "comparisons": [compare(args.reference, path, cross_backend=args.cross_backend) for path in args.candidate]
    }
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
