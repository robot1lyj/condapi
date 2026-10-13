"""Derive immutable return-preserving decision/wait replay for offline comparisons."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.learning_view import learning_view


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--keep-wait", action="store_true")
    args = p.parse_args()
    parent = json.loads((args.replay / "manifest.json").read_text())
    if sha256(args.replay / "replay.npz") != parent["replay_sha256"]:
        raise ValueError("parent replay changed")
    with np.load(args.replay / "replay.npz", allow_pickle=False) as z:
        arrays, audit = learning_view({k: z[k] for k in z.files}, collapse_wait=not args.keep_wait)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "replay.npz", **arrays)
    write_json(args.output / "audit.json", audit)
    write_json(args.output / "manifest.json", {"schema": "pld_learning_replay_v1", "contract": parent["contract"],
        "snapshots": parent["snapshots"], "parent_manifest_sha256": sha256(args.replay / "manifest.json"),
        "parent_replay_sha256": parent["replay_sha256"], "parent": str(args.replay),
        "replay_sha256": sha256(args.output / "replay.npz"), "audit_sha256": sha256(args.output / "audit.json"),
        "learning_view_source_sha256": sha256(Path(__file__).resolve().parents[2] / "packages/pld-rl/src/yam_pld_rl/learning_view.py"),
        "reward_contract_unchanged": True, "discounted_return_preserved": True,
        "collapse_wait": not args.keep_wait})
    print(json.dumps({"native": len(next(iter(arrays.values()))) + sum(a["merged_wait_intervals"] for a in audit["attempts"]),
                      "learning": len(arrays["reward"]), "attempts": len(audit["attempts"])}))


if __name__ == "__main__":
    main()
