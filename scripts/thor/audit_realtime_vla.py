"""Compare saved arrays and export reproducible evidence without GPU execution."""

import argparse
import json
from pathlib import Path
import shutil

from compare_suites import digest
from compare_suites import error_metrics
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    runs = args.runs_root
    evidence = args.evidence
    a_path = args.old_results / "pi05-A-20260907-r2/actions.npy"
    old_w_path = args.old_results / "pi05-W-20260908-r1/actions.npy"
    new_w_path = runs / "pi05-W-20260911-realtime-control/actions.npy"
    position = runs / "realtime-vla-position-20260911-r1/actions.npy"
    time_fix = runs / "realtime-vla-time-position-20260911-r1/actions.npy"
    new_w = np.load(new_w_path, allow_pickle=False)
    comparison = {
        "W_vs_JAX": error_metrics(np.load(a_path, allow_pickle=False)[:, 0], new_w[:, 0]),
        "W_exact_vs_20260908": bool(np.array_equal(new_w, np.load(old_w_path, allow_pickle=False))),
        "time_broadcast_exact_vs_position_only": bool(
            np.array_equal(np.load(position, allow_pickle=False), np.load(time_fix, allow_pickle=False))
        ),
        "array_sha256": {str(p): digest(p) for p in (a_path, old_w_path, new_w_path, position, time_fix)},
        "excluded_runs": {"realtime-vla-upstream-20260911-r1": "harness omitted image normalization; invalid"},
        "scope": "offline base-model replay; no robot accuracy acceptance",
    }
    for p in runs.glob("*/result.json"):
        shutil.copyfile(p, evidence / f"{p.parent.name}.result.json")
    (evidence / "array-audit.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
