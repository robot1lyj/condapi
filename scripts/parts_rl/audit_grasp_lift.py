"""Gate existing visual proposals on post-closure lift evidence; CPU/stdlib only.

Writes a new sidecar directory, never overwrites recordings, proposals or reviews.
The gate is necessary evidence eligibility, not an automatic grasp classifier.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_reviews(data, candidates, identity):
    if data.get("schema") != "revised_grasp_human_review_v3" or data.get("identity") != identity:
        raise ValueError("Review schema/identity does not match immutable candidates")
    records = data["records"]
    ids = [r["candidate_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != {c["candidate_id"] for c in candidates}:
        raise ValueError("Review must contain each candidate exactly once")
    if any(r.get("review_label") not in (None, "success", "failure", "non_grasp") for r in records):
        raise ValueError("Unsupported review label")
    return {r["candidate_id"]: r for r in records}


def lift_evidence(candidate, trace, root, threshold=0.05):
    """Use closure as reference, only before release, with fresh same-arm images."""
    close, end = candidate["close_row"], candidate["inspection_end_row"]
    xyz = trace["arms"][candidate["arm"]]["position_m"]
    times = trace["time"]
    if not 0 <= close <= end < len(xyz):
        raise ValueError("Invalid inspection interval")
    reference = xyz[close][2]
    if reference is None or not math.isfinite(reference):
        raise ValueError("Missing closure FK reference")
    finite_rows = [r for r in range(close, end + 1) if xyz[r][2] is not None and math.isfinite(xyz[r][2])]
    max_row = max(finite_rows, key=lambda r: xyz[r][2])
    crossed = [r for r in finite_rows if xyz[r][2] - reference >= threshold]
    qualified = []
    image_ids = set()
    for frame in candidate["frames"]:
        row = frame["row"]
        if not close <= row <= end or row not in crossed:
            continue
        view = frame["views"].get(candidate["arm"])
        if not view or not 0 <= times[row] - view["capture_time"] <= 0.3:
            continue
        if not (root / view["image"]).is_file():
            raise ValueError(f"Missing evidence image: {view['image']}")
        image_key = (view["image"], view["video_frame_index"])
        if image_key not in image_ids:
            image_ids.add(image_key)
            qualified.append(row)
    return {
        "reference_row": close,
        "reference_base_z_m": reference,
        "inspection_end_row": end,
        "max_postclosure_rise_m": xyz[max_row][2] - reference,
        "max_rise_row": max_row,
        "first_threshold_row": crossed[0] if crossed else None,
        "selected_lift_rows": qualified,
        "has_lift_evidence": bool(qualified),
        "single_selected_lift_frame": len(qualified) == 1,
        "height_semantics": "recorded_base_z_relative_to_closure_not_calibrated_table_height",
    }


def proposal(candidate, review, evidence):
    label = review.get("review_label") or candidate["suggested_label"]
    eligible = evidence["has_lift_evidence"] and label in ("success", "failure")
    return {
        "candidate_id": candidate["candidate_id"],
        "episode_number": candidate["episode_number"],
        "arm": candidate["arm"],
        "split": candidate["split"],
        "source_label": candidate["suggested_label"],
        "review_label": review.get("review_label"),
        "reviewed_at": review.get("reviewed_at"),
        "notes": review.get("notes", ""),
        "effective_label": label,
        "label_source": "user_review" if review.get("review_label") else candidate["label_source"],
        "lift_evidence": evidence,
        "evidence_eligible": eligible,
        "exclusion_reason": (
            "user_non_grasp" if label == "non_grasp"
            else "missing_post_lift_verification" if not evidence["has_lift_evidence"]
            else None
        ),
        "proposed_reward": (1 if label == "success" else 0) if eligible else None,
        "approved_reward": None,
        "ready_for_training": False,
    }


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = (args.dataset / "candidates.json").read_bytes()
    review_bytes = args.human_review.read_bytes()
    candidates = json.loads(source_bytes)
    if len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate source candidates")
    reviews = load_reviews(json.loads(review_bytes), candidates, sha(source_bytes))
    traces, trace_hashes = {}, {}
    rows = []
    for candidate in candidates:
        ep = candidate["episode_number"]
        if ep not in traces:
            data = (args.dataset / "traces" / f"E{ep:02d}.json").read_bytes()
            traces[ep], trace_hashes[f"E{ep:02d}"] = json.loads(data), sha(data)
        rows.append(proposal(candidate, reviews[candidate["candidate_id"]],
                             lift_evidence(candidate, traces[ep], args.dataset)))
    counts = Counter(r["effective_label"] for r in rows if r["evidence_eligible"])
    summary = {
        "schema": "grasp_lift_qualification_v1",
        "dataset_id": args.output.name,
        "source_dataset": str(args.dataset.resolve()),
        "source_candidates_sha256": sha(source_bytes),
        "human_review_sha256": sha(review_bytes),
        "trace_sha256": trace_hashes,
        "minimum_relative_base_z_rise_m": 0.05,
        "candidates": len(rows),
        "user_reviewed": sum(bool(r["review_label"]) for r in rows),
        "qualified_visual_proposals": dict(counts),
        "excluded_ids": [r["candidate_id"] for r in rows if not r["evidence_eligible"]],
        "single_selected_lift_frame_ids": [r["candidate_id"] for r in rows
                                           if r["lift_evidence"]["single_selected_lift_frame"]],
        "approved_rewards": 0,
        "ready_for_training": False,
        "neural_inference": False,
        "note": "Eligibility is necessary, not proof of correct labels; original artifacts remain unchanged.",
    }
    args.output.mkdir(exist_ok=False)
    (args.output / "human-review-source.json").write_bytes(review_bytes)
    write(args.output / "lift-qualified-proposals.json", rows)
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
