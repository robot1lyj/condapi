"""Publish model suggestions plus human overrides against a frozen curator release.

Reuse audited segmentation/execution, never rerun FK or alter candidate identities.
This is CPU data preparation, not a learning loop.
"""

import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.curator import associate_rewards
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.review_policy import merge_rewards


def row_index(folder):
    import h5py  # noqa: PLC0415

    rows = []
    for path in sorted(folder.glob("segment_*/samples.h5")):
        with h5py.File(path) as f:
            n = int(f["committed_rows"][()])
            rows.extend({"tick": int(tick), "raw_ref": f"{folder.parent.name}/{folder.name}/{path.parent.name}/samples.h5#row={i}"}
                        for i, tick in enumerate(f["tick"][:n]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--curator", default="curator-v2")
    p.add_argument("--vision", default="vision-v1")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    parent = args.dataset / args.curator
    manifest = json.loads((parent / "manifest.json").read_text())
    for name, digest in manifest["outputs_sha256"].items():
        if sha256(parent / name) != digest:
            raise ValueError("parent evidence modified: " + name)
    for name, digest in manifest["sources_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("raw source modified: " + name)
    candidates = json.loads((parent / "candidates.json").read_text())
    judged_path = args.dataset / args.vision / "candidates.json"
    judged = {c["candidate_id"]: c for c in json.loads(judged_path.read_text())}
    state_path = args.dataset / "reviews/user-review-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"identity": sha256(parent / "candidates.json"), "reviews": {}}
    if state["identity"] != sha256(parent / "candidates.json"):
        raise ValueError("human reviews belong to different candidate release")
    merged = merge_rewards(candidates, judged, state["reviews"])
    approvals = {r["candidate_id"]: r for r in merged["rewards"]}
    episodes = {}
    for item in json.loads((parent / "audit.json").read_text()):
        folder = Path(item["episode"])
        eid = json.loads((folder / "manifest.json").read_text())["episode_id"]
        episodes[eid] = (folder, item["split"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "approved-labels.json", merged)
    labels, audits, seen = [], [], set()
    count = 0
    with (parent / "windows.jsonl").open() as source, (args.output / "records.jsonl").open("w") as target:
        records = (json.loads(line) for line in source if line.strip())
        for eid, group in groupby(records, key=lambda r: r["proposal"]["episode_id"]):
            if eid in seen:
                raise ValueError("parent windows must be grouped by episode")
            seen.add(eid)
            folder, split = episodes[eid]
            cs = [c for c in candidates if c["episode_id"] == eid]
            rs = [approvals[c["candidate_id"]] for c in cs if c["candidate_id"] in approvals]
            linked, rewards, exclusions = associate_rewards(list(group), cs, rs, row_index(folder), split)
            for reward in rewards:
                proposal = approvals[reward["attempt_id"]]
                reward["label_provenance"] = {k: proposal[k] for k in (
                    "candidate_sha256", "label_source", "human_reviewed", "model_confidence", "model_audit_flags",
                    "qwen_suggestion", "dino_suggestion", "note") if k in proposal}
            for record in linked:
                target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            count += len(linked)
            labels.extend(rewards)
            audits.append({"episode": str(folder), "split": split, "associated_windows": len(rewards),
                           "actual_intervals": len(linked), "association_exclusions": exclusions})
            print(json.dumps({"episode": folder.name, "windows": len(rewards), "intervals": len(linked)}), flush=True)
    write_json(args.output / "rewards.json", labels)
    write_json(args.output / "audit.json", audits)
    summary = {"windows": len(labels), "intervals": count, "statuses": dict(Counter(r["status"] for r in labels)),
               "splits": dict(Counter(r["split"] for r in labels)), "pending": len(merged["pending"]),
               "human_saved": len(state["reviews"]), "human_valid": sum(r["human_reviewed"] for r in merged["rewards"]),
               "policy": merged["policy"], "robot_success_rate": None, "detection_accuracy": None}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "manifest.json", {"schema": "pld_curator_v2", "producer": "Replay-Curator",
               "ready_for_replay_compilation": bool(count), "purpose": "automatic_suggestion_reward_experiment",
               "parent_manifest_sha256": sha256(parent / "manifest.json"), "parent_manifest": manifest,
               "judgment_sha256": sha256(judged_path), "human_state_sha256": sha256(state_path) if state_path.exists() else None,
               "producer_files_sha256": {str(Path(__file__).name): sha256(Path(__file__)),
                                         "review_policy.py": sha256(Path(__file__).resolve().parents[2] / "packages/pld-rl/src/yam_pld_rl/review_policy.py"),
                                         "curator.py": sha256(Path(__file__).resolve().parents[2] / "packages/pld-rl/src/yam_pld_rl/curator.py")},
               "outputs_sha256": {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()},
               "robot_success_rate": None})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
