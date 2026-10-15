"""Freeze same-replay small-network comparisons; CPU data preparation only."""
# ruff: noqa: E402

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-prefix", required=True)
    args = p.parse_args()
    replay = args.dataset / "learning-replay-v1"
    manifest = json.loads((replay / "manifest.json").read_text())
    if sha256(replay / "replay.npz") != manifest["replay_sha256"]:
        raise ValueError("replay changed")
    behavior = args.dataset / "behavior-bundle/manifest.json"
    bm = json.loads(behavior.read_text())
    if manifest["contract"] != bm["contract"]:
        raise ValueError("behavior contract changed")
    with np.load(replay / "replay.npz", allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}
    train = z["train"].astype(bool)
    if not train.any() or train.all():
        raise ValueError("both episode-level training and validation required")
    rms = [float(np.sqrt(np.mean(np.sum(z["x"][train, a:b] ** 2, axis=1))))
           for a, b in ((0, 1536), (1536, 1550), (1550, 2250), (2250, 2264), (2264, 2287))]
    if not all(np.isfinite(rms)) or min(rms) <= 0:
        raise ValueError("invalid feature scales")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configs").mkdir()
    jobs = []
    templates = {"td3bc": "parts_td3bc", "iql": "iql", "sac": "sac", "bc": "parts_td3bc"}
    for name, method in templates.items():
        for seed in (0, 1):
            cfg = json.loads((ROOT / f"configs/pld_rl/continuous10-r04/pld-ct-{name}-r04-s{seed}.json").read_text())
            run = f"{args.run_prefix}-{name}-s{seed}"
            cfg.update(id=run, method=method, feature_group_rms=rms,
                       feature_balance_source={"replay_sha256": manifest["replay_sha256"],
                                               "training_rows": int(train.sum()), "validation_used": False},
                       behavior_snapshot_sha256=sha256(behavior))
            write_json(args.output / "configs" / (run + ".json"), cfg)
            jobs.append(run)
    write_json(args.output / "contract.json", manifest["contract"])
    write_json(args.output / "normalization.json", bm["normalization"])
    summary = {"jobs": jobs, "replay_sha256": manifest["replay_sha256"],
               "behavior_snapshot_sha256": sha256(behavior), "feature_group_rms": rms,
               "decisions": len(train), "train_decisions": int(train.sum()),
               "val_decisions": int((~train).sum()), "training_executed": False,
               "producer_sha256": sha256(Path(__file__)), "old_contract_replay_mixed": False}
    for split, mask in (("train", train), ("val", ~train)):
        attempts = np.unique(z["attempt"][mask])
        outcomes = [int(z["attempt_outcome"][np.flatnonzero(z["attempt"] == a)[0]]) for a in attempts]
        summary[split + "_attempts"] = len(attempts)
        summary[split + "_outcomes"] = dict(Counter(outcomes))
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
