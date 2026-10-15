"""Fuse frozen VLM decisions with independent DINO evidence and explicit audit corrections."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def dino_predictions(root):
    jobs_path = root / "feature-jobs.json"
    jobs = read(jobs_path)
    run = read(root / "dinov3/run.json")
    if run["jobs_sha256"] != hashlib.sha256(jobs_path.read_bytes()).hexdigest():
        raise ValueError("Feature jobs do not match inference run")
    with np.load(root / "dinov3/features.npz") as saved:
        if saved["ids"].tolist() != [j["id"] for j in jobs]:
            raise ValueError("DINO feature order mismatch")
        features = {
            k: saved[k] / np.maximum(np.linalg.norm(saved[k], axis=1, keepdims=True), 1e-12) for k in ["roi", "cls"]
        }
    refs = [(i, j) for i, j in enumerate(jobs) if j["role"] == "reference"]
    output = {}
    for i, job in enumerate(jobs):
        if job["role"] != "query":
            continue
        groups = {
            label: [
                ri
                for ri, r in refs
                if r["label"] == label
                and r["episode_number"] != job["episode_number"]
                and r.get("source_sha256") != job.get("source_sha256")
            ]
            for label in ["held_at_snapshot", "empty_at_snapshot"]
        }
        if not all(groups.values()):
            raise ValueError("Reference bank must retain both labels after episode exclusion")
        record = {"id": job["id"], "excluded_episode": job["episode_number"]}
        for branch, f in features.items():
            nearest = {}
            for label, indexes in groups.items():
                scores = f[indexes] @ f[i]
                ri = indexes[int(np.argmax(scores))]
                nearest[label] = {"id": jobs[ri]["id"], "similarity": float(np.max(scores))}
            record[branch] = {
                "margin": nearest["held_at_snapshot"]["similarity"] - nearest["empty_at_snapshot"]["similarity"],
                "nearest": nearest,
            }
        output[job["id"]] = record
    return output


def summarize_dino(candidate, frames):
    rows = []
    for f in candidate["frames"]:
        if f["since_close_s"] < 0.45 or candidate["arm"] not in f["views"]:
            continue
        q = frames[f["views"][candidate["arm"]]["key"]]
        margin = (q["roi"]["margin"] + q["cls"]["margin"]) / 2
        rows.append(
            {
                "row": f["row"],
                "since_close_s": f["since_close_s"],
                "lift_from_closure_m": f["lift_from_closure_m"],
                "roi_margin": q["roi"]["margin"],
                "cls_margin": q["cls"]["margin"],
                "average_margin": margin,
            }
        )
    if not rows:
        return {"outcome": None, "reason": "no_post_closure_snapshot", "frames": []}
    # Later carried-object evidence is useful, but never turns height into a success label.
    strongest = sorted(rows, key=lambda q: q["average_margin"], reverse=True)
    support = strongest[:2]
    success = len(support) == 2 and all(q["roi_margin"] > 0.02 and q["cls_margin"] > 0.02 for q in support)
    clear_empty = all(q["roi_margin"] < -0.02 and q["cls_margin"] < -0.02 for q in rows)
    return {
        "outcome": "success" if success else "failure" if clear_empty else None,
        "reason": "two_held_snapshots"
        if success
        else "all_selected_snapshots_empty"
        if clear_empty
        else "mixed_or_weak_snapshot_evidence",
        "frames": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--vlm", default="vlm-primary")
    p.add_argument("--audit", type=Path)
    args = p.parse_args()
    root = args.dataset
    frames = dino_predictions(root)
    audits = {q["candidate_id"]: q for q in read(args.audit)} if args.audit else {}
    candidates = read(root / "candidates-prepared.json")
    results = []
    for c in candidates:
        raw = read(root / args.vlm / (c["candidate_id"] + ".json"))
        if raw["montage_sha256"] != c["montage_sha256"] or raw["parse_error"] or not raw["decision"]:
            raise ValueError("Missing, invalid or stale VLM evidence: " + c["candidate_id"])
        decision = raw["decision"]
        dino = summarize_dino(c, frames)
        flags = list(decision.get("format_audit_flags", []))
        if decision["confidence"] != "high":
            flags.append("vlm_" + decision["confidence"] + "_confidence")
        if dino["outcome"] and dino["outcome"] != decision["outcome"]:
            flags.append("vlm_dino_disagreement")
        if not dino["outcome"]:
            flags.append("dino_weak_or_mixed")
        c.update(vlm=raw, dino=dino, visual_audit=audits.get(c["candidate_id"]))
        corrected = audits.get(c["candidate_id"], {})
        event_type = corrected.get("event_type", decision["event_type"])
        outcome = corrected.get("outcome", decision["outcome"])
        if corrected and (event_type != decision["event_type"] or outcome != decision["outcome"]):
            flags.append("vlm_visual_audit_disagreement")
        if corrected.get("confidence") in ["medium", "low"]:
            flags.append("visual_audit_" + corrected["confidence"] + "_confidence")
        if event_type not in ["grasp", "non_grasp"] or outcome not in ["success", "failure"]:
            raise ValueError("Audit has invalid categorical label")
        c.update(
            event_type=event_type,
            suggested_label=outcome if event_type == "grasp" else "non_grasp",
            proposed_reward=(1 if outcome == "success" else 0) if event_type == "grasp" else None,
            confidence=corrected.get("confidence", decision["confidence"]),
            audit_flags=flags,
            label_source="assistant_visual_audit" if corrected else "frozen_vlm",
            reasons=[corrected.get("reason", decision["reason"])],
            review_label=None,
            approved_reward=None,
        )
        results.append(c)
    summary = {
        "schema": "revised_grasp_binary_v3",
        "dataset_id": root.name,
        "episodes": len({q["episode_number"] for q in results}),
        "candidates": len(results),
        "suggested_counts": dict(Counter(q["suggested_label"] for q in results)),
        "confidence_counts": dict(Counter(q["confidence"] for q in results)),
        "audit_flag_counts": dict(Counter(f for q in results for f in q["audit_flags"])),
        "assistant_visually_audited": len(audits),
        "user_reviewed": 0,
        "approved_rewards": 0,
        "automatic_tracking": False,
        "training_executed": False,
        "accuracy": None,
        "accuracy_note": "No independent human-labelled test set; binary coverage is not accuracy",
        "references": sum(q["role"] == "reference" for q in read(root / "feature-jobs.json")),
        "same_episode_references_excluded": True,
        "preparation": read(root / "preparation.json"),
        "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for name, data in [("frame-classifications.json", frames), ("candidates.json", results), ("summary.json", summary)]:
        (root / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    columns = [
        "candidate_id",
        "episode_number",
        "arm",
        "close_row",
        "inspection_end_row",
        "end_row",
        "event_type",
        "suggested_label",
        "proposed_reward",
        "confidence",
        "label_source",
        "review_label",
        "approved_reward",
    ]
    with (root / "classification.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
