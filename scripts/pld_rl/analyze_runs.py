"""Compare entry return estimates with simple baselines, using saved predictions."""

import argparse
import json
from pathlib import Path

import numpy as np


def auc(scores, outcomes):
    positive, negative = scores[outcomes == 1], scores[outcomes == 0]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return float(((differences > 0) + 0.5 * (differences == 0)).mean())


def analyze(root):
    result = {"scope": "entry predictions only; not robot success rate", "runs": []}
    for config_path in sorted((root / "configs").glob("*.json")):
        config = json.loads(config_path.read_text())
        folder = root / config["id"]
        status = json.loads((folder / "status.json").read_text())
        if status["state"] != "completed":
            raise ValueError("analysis requires completed runs")
        with np.load(folder / f"attempt-predictions-{status['step']:06d}.npz") as source:
            p = {key: source[key] for key in source.files}
        score = np.minimum(p["q1"], p["q2"])
        valid = p["mc_valid"] & p["entry_precedes_closure"]
        train = valid & (p["episode"] != 1)
        if config.get("success_only", False):
            train &= p["outcome"] == 1
        val = valid & (p["episode"] == 1)
        train_mean = float(p["mc_return"][train].mean())
        target = p["mc_return"][val]
        item = {
            "id": config["id"],
            "step": status["step"],
            "validation_attempts": int(val.sum()),
            "rmse": float(np.sqrt(np.mean((score[val] - target) ** 2))),
            "auc": auc(score[val], p["outcome"][val]),
            "constant_one_rmse": float(np.sqrt(np.mean((1 - target) ** 2))),
            "train_mean_prediction": train_mean,
            "train_mean_rmse": float(np.sqrt(np.mean((train_mean - target) ** 2))),
            "arms": {},
        }
        for arm, name in [(0, "left"), (1, "right")]:
            selected = val & (p["arm"] == arm)
            arm_result = {
                "attempts": int(selected.sum()),
                "successes": int(np.sum(p["outcome"][selected] == 1)),
                "failures": int(np.sum(p["outcome"][selected] == 0)),
                "auc": auc(score[selected], p["outcome"][selected]),
            }
            for outcome, label in [(0, "empty"), (1, "held")]:
                mask = selected & (p["outcome"] == outcome)
                arm_result[f"{label}_mean_q"] = float(score[mask].mean()) if mask.any() else None
            item["arms"][name] = arm_result
        result["runs"].append(item)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("keep the original analysis; use a new output version")
    args.output.write_text(json.dumps(analyze(args.run_root), indent=2, allow_nan=False) + "\n")
