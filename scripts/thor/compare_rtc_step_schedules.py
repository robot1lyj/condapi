"""Compare original-JAX RTC solvers with identical checkpoint, cases and noise."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

def digest(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_reference(directory: Path):
    manifest_path = directory / "reference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "original_jax_trained_rtc_reference":
        raise ValueError("Expected original-JAX RTC reference")
    arrays = []
    for row in manifest["cases"]:
        path = directory / row["reference"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Reference changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            physical = data["physical"].copy()
        if physical.shape != (50, 14) or not np.isfinite(physical).all():
            raise ValueError("Invalid physical H50/14D reference")
        arrays.append(physical)
    return manifest, np.stack(arrays)


def stats(values):
    return {
        "max_abs": float(values.max()),
        "mean_abs": float(values.mean()),
        "p95_abs": float(np.percentile(values, 95)),
        "p99_abs": float(np.percentile(values, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new comparison output path")
    base_manifest, baseline = load_reference(args.baseline)
    candidate_manifest, candidate = load_reference(args.candidate)
    keys = (
        "checkpoint_metadata_sha256", "source_params_files_sha256",
        "training_contract_sha256", "norm_stats_sha256", "cases_sha256", "noise_seed",
    )
    if any(base_manifest.get(key) != candidate_manifest.get(key) for key in keys):
        raise ValueError("Step schedules do not share checkpoint/norm/cases/noise")
    if [row["sample"] for row in base_manifest["cases"]] != [
        row["sample"] for row in candidate_manifest["cases"]
    ] or [row["delay_steps"] for row in base_manifest["cases"]] != [
        row["delay_steps"] for row in candidate_manifest["cases"]
    ]:
        raise ValueError("Step schedules have different case order or delays")
    error = np.abs(candidate.astype(np.float64) - baseline.astype(np.float64))
    joint_indices = [i for i in range(14) if i not in (6, 13)]
    first_generated = np.stack([
        error[index, row["delay_steps"]:row["delay_steps"] + 10]
        for index, row in enumerate(candidate_manifest["cases"])
    ])
    worst = np.unravel_index(int(error.argmax()), error.shape)
    result = {
        "status": "solver_schedule_difference_not_task_accuracy",
        "observed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "baseline_steps": base_manifest.get("num_steps", 10),
        "candidate_steps": candidate_manifest.get("num_steps", 10),
        "baseline_manifest_sha256": digest(args.baseline / "reference_manifest.json"),
        "candidate_manifest_sha256": digest(args.candidate / "reference_manifest.json"),
        "all_14d": stats(error),
        "joints_rad": stats(error[:, :, joint_indices]),
        "grippers_normalized": stats(error[:, :, [6, 13]]),
        "first_10_actions_14d": stats(error[:, :10]),
        "first_10_generated_actions_14d": stats(first_generated),
        "worst_element": {
            "case_index": int(worst[0]), "action_step": int(worst[1]), "dimension": int(worst[2])
        },
        "per_case": [
            {"sample": row["sample"], "delay_steps": row["delay_steps"], **stats(error[index])}
            for index, row in enumerate(candidate_manifest["cases"])
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_case"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
