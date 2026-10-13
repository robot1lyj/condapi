"""CPU-only audit of recorded correction timing and feature-block energy."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    z = np.load(args.dataset/"learning-replay-r02-v1/replay.npz", allow_pickle=False)
    wanted, wins = set(z["attempt"]), {}
    with (args.dataset/"reward-release-v2/records.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            proposal = record["proposal"]
            key = "/".join((proposal["run_id"], proposal["episode_id"], proposal["attempt_id"]))
            if key not in wanted:
                continue
            w = wins.setdefault(key, {"close": None, "effects": set(), "arm": proposal["arm"]})
            if proposal["gate"].get("close_tick") is not None:
                w["close"] = proposal["gate"]["close_tick"]
            for effect in record["receipt"].get("actor_effects_before_handoff", []):
                w["effects"].add(effect["tick"])
    if set(wins) != wanted:
        raise ValueError("missing eligible attempts")
    lead, before, missing = [], [], 0
    for w in wins.values():
        if w["close"] is None or not w["effects"]:
            missing += 1
            continue
        lead.append((w["close"]-min(w["effects"]))/30)
        before.append(sum(t<w["close"] for t in w["effects"]))
    x = z["x"][z["train"].astype(bool)]
    parts = {"visual": (0, 1536), "state": (1536, 1550), "plan": (1550, 2250),
             "previous_command": (2250, 2264), "context_and_limits": (2264, 2287)}
    energy = {k: {"mean_squared_l2": float(np.mean(np.sum(x[:, a:b]**2, axis=1))),
                  "mean_squared_centered_l2": float(np.mean(np.sum((x[:, a:b]-x[:, a:b].mean(0))**2, axis=1)))}
              for k, (a, b) in parts.items()}
    result = {"eligible_windows": len(wins), "missing_close_or_effect": missing,
              "closure_is_online_relative_gripper_detector_not_contact_ground_truth": True,
              "first_effect_to_closure_seconds_quantiles": np.quantile(lead, [0, .25, .5, .75, 1]).tolist(),
              "no_effect_before_detected_closure": sum(v == 0 for v in before),
              "effect_ticks_before_closure_quantiles": np.quantile(before, [0, .25, .5, .75, 1]).tolist(),
              "train_feature_energy": energy, "interpretation": "energy alone does not prove modality use or causality"}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
