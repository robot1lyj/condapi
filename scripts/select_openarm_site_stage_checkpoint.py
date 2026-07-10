"""Select a Site-Stage checkpoint that improves Site without forgetting HQ Stage ordering."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

SITE_GATES = {
    "mse": ("max", 0.020),
    "mae": ("max", 0.110),
    "sign_accuracy": ("min", 0.88),
    "corrcoef": ("min", 0.92),
    "r2": ("min", 0.80),
}
HQ_GATES = {
    "mse": ("max", 0.018),
    "mae": ("max", 0.100),
    "sign_accuracy": ("min", 0.88),
    "corrcoef": ("min", 0.93),
    "r2": ("min", 0.84),
}


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _apply_gates(metrics: dict[str, Any], gates: dict[str, tuple[str, float]]) -> dict[str, bool]:
    results = {}
    for name, (direction, threshold) in gates.items():
        value = metrics.get(name)
        if value is None:
            results[f"{name}_{direction}_{threshold}"] = False
        elif direction == "max":
            results[f"{name}<={threshold}"] = float(value) <= threshold
        else:
            results[f"{name}>={threshold}"] = float(value) >= threshold
    return results


def select_checkpoint(
    report_dir: pathlib.Path,
    checkpoint_root: pathlib.Path,
    steps: list[int],
) -> dict[str, Any]:
    candidates = []
    for step in steps:
        site_path = report_dir / f"site_{step}.json"
        hq_path = report_dir / f"hq_{step}.json"
        if not site_path.exists() or not hq_path.exists():
            raise FileNotFoundError(f"Missing evaluation reports for checkpoint {step}")
        site = _load_json(site_path)
        hq = _load_json(hq_path)
        site_gates = _apply_gates(site, SITE_GATES)
        hq_gates = _apply_gates(hq, HQ_GATES)
        checkpoint = checkpoint_root / str(step)
        passed = checkpoint.is_dir() and all(site_gates.values()) and all(hq_gates.values())
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "site": site,
                "hq": hq,
                "site_gates": site_gates,
                "hq_gates": hq_gates,
                "passed": passed,
            }
        )

    passing = [candidate for candidate in candidates if candidate["passed"]]
    if not passing:
        raise ValueError("No Site-Stage checkpoint passed both Site and HQ retention gates")
    best = min(
        passing,
        key=lambda candidate: (
            float(candidate["site"]["mse"]),
            -float(candidate["site"]["r2"]),
            float(candidate["hq"]["mse"]),
        ),
    )
    return {
        "selected_step": best["step"],
        "selected_checkpoint": best["checkpoint"],
        "selection_rule": "lowest Site MSE among checkpoints passing Site quality and HQ retention gates",
        "site_gates": SITE_GATES,
        "hq_gates": HQ_GATES,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-root", type=pathlib.Path, required=True)
    parser.add_argument("--steps", default="1000,2000,3000,4000,4999")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    result = select_checkpoint(args.report_dir, args.checkpoint_root, steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
