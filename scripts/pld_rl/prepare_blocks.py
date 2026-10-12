"""CPU-only preparation of actual PLD decisions; never changes source recordings."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.block_replay import compile_replay
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, nargs="+", required=True, help="joined proposal/execution JSONL")
    p.add_argument("--rewards", type=Path, required=True)
    p.add_argument("--bundles", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    snapshots = {sha256(b / "manifest.json"): json.loads((b / "manifest.json").read_text()) for b in args.bundles}
    contract = next(iter(snapshots.values()))["contract"]
    records = [json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
    rewards = json.loads(args.rewards.read_text())
    data, audit = compile_replay(records, rewards, contract, snapshots)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "replay.npz", **data)
    manifest = {"schema": "pld_sac_replay_v1", "contract": contract, "snapshots": snapshots, "audit": audit,
                "replay_sha256": sha256(args.output / "replay.npz"), "reward_sha256": sha256(args.rewards),
                "records_sha256": {str(p): sha256(p) for p in args.records}}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "rewards.json", rewards)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
