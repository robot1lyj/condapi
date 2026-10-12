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
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--records", type=Path, nargs="+", help="derived proposal/execution JSONL, not client output")
    group.add_argument("--curator", type=Path, help="immutable reviewed curate_recording.py output release")
    p.add_argument("--rewards", type=Path)
    p.add_argument("--bundles", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    curator = None
    if args.curator:
        if args.rewards:
            p.error("--curator already binds rewards; do not override them")
        curator = json.loads((args.curator / "manifest.json").read_text())
        if curator["schema"] != "pld_curator_v2" or not curator["ready_for_replay_compilation"]:
            raise ValueError("curator release has no reviewed windows")
        for name, digest in curator["outputs_sha256"].items():
            if sha256(args.curator / name) != digest:
                raise ValueError("curator evidence/output modified after release")
        args.records = [args.curator / "records.jsonl"]
        args.rewards = args.curator / "rewards.json"
    elif not args.rewards:
        p.error("--records requires --rewards")
    snapshots = {sha256(b / "manifest.json"): json.loads((b / "manifest.json").read_text()) for b in args.bundles}
    contract = next(iter(snapshots.values()))["contract"]
    records = [json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
    rewards = json.loads(args.rewards.read_text())
    data, audit = compile_replay(records, rewards, contract, snapshots)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "replay.npz", **data)
    manifest = {"schema": "pld_sac_replay_v2", "contract": contract, "snapshots": snapshots, "audit": audit,
                "replay_sha256": sha256(args.output / "replay.npz"), "reward_sha256": sha256(args.rewards),
                "records_sha256": {str(p): sha256(p) for p in args.records}}
    if curator is not None:
        manifest["curator"] = curator
        manifest["curator_manifest_sha256"] = sha256(args.curator / "manifest.json")
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "rewards.json", rewards)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
