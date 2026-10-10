"""Prepare all existing grasp candidates for frozen, multi-snapshot classification."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def eligible(candidate, sample):
    capture = sample["visual_time"]
    return bool(
        sample["row"] >= candidate["close_row"]
        and sample["row"] <= candidate["end_row"]
        and sample["valid"]
        and sample["openness"] < 0.6
        and sample["lift_delta_m"] is not None
        and sample["lift_delta_m"] >= 0.05
        and capture is not None
        and 0 <= sample["time"] - capture <= 0.25
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    root.mkdir(exist_ok=False)
    for sub in ["images", "clips", "reference", "reference/crops", "run-scripts"]:
        (root / sub).mkdir(exist_ok=False)
    source_jobs = {j["frame_key"]: j for j in json.loads((args.source / "image-jobs.json").read_text())}
    candidates = json.loads((args.source / "candidates.json").read_text())
    ids = [c["candidate_id"] for c in candidates]
    if len(set(ids)) != len(ids) or len(candidates) != 273:
        raise ValueError("expected the complete 273-candidate source")
    jobs, seen, prepared = [], set(), []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        samples = [{k: v for k, v in s.items() if k not in ["vision", "held_lifted", "empty_lifted"]}
                   for s in candidate["samples"]]
        selected = [s for s in samples if eligible(candidate, s)]
        # Retain an illustrative frame even when no lifted frame can be classified.
        display = selected or [max(samples, key=lambda s: s["lift_delta_m"] if s["lift_delta_m"] is not None else -1e9)]
        for sample in display:
            key = sample["frame_key"]
            image = args.source / sample["image"]
            identity = source_jobs[key]["sha256"]
            if sha(image) != identity:
                raise ValueError("source image identity mismatch")
            sample["image"] = f"images/{key}.jpg"
            shutil.copyfile(image, root / sample["image"])
            if not selected:
                continue
            if key in seen:
                raise ValueError("duplicate frame assigned to candidates")
            seen.add(key)
            jobs.append({"id": key, "candidate_id": cid, "image": sample["image"], "sha256": identity,
                         "view": "wrist_full", "arm": candidate["arm"], "episode_number": candidate["episode_number"],
                         "source_frame_key": key, "row": sample["row"], "time": sample["time"],
                         "visual_time": sample["visual_time"], "epoch": sample["epoch"],
                         "camera_sequence": sample["camera_sequence"], "lift_delta_m": sample["lift_delta_m"],
                         "openness": sample["openness"], "clip": candidate["review_clip"]["path"],
                         "approved_reward": None})
        clip = candidate["review_clip"]
        if sha(args.source / clip["path"]) != clip["sha256"]:
            raise ValueError("source clip identity mismatch")
        shutil.copyfile(args.source / clip["path"], root / clip["path"])
        record = {k: v for k, v in candidate.items() if k not in [
            "samples", "suggested_label", "reasons", "visual_hold_s", "clear_empty_s", "review_gallery"]}
        record.update(samples=samples, eligible_frame_ids=[s["frame_key"] for s in selected],
                      fallback_image=None if selected else display[0]["image"],
                      review_label=None, approved_reward=None, review_status="unreviewed")
        prepared.append(record)
    references = [j for j in json.loads((args.reference / "feature-jobs.json").read_text()) if j["split"] == "reference"]
    label_path = args.reference / "reference-labels.json"
    labels = json.loads(label_path.read_text())
    if {r["candidate_id"] for r in labels["records"]} != {j["id"] for j in references}:
        raise ValueError("reference labels do not match reference bank")
    feature_path = args.reference / "dinov3/features.npz"
    with np.load(feature_path) as features:
        by_id = {cid: i for i, cid in enumerate(features["ids"].tolist())}
        indices = [by_id[j["id"]] for j in references]
        np.savez_compressed(root / "reference/features.npz", ids=[j["id"] for j in references],
                            roi=features["roi"][indices], cls=features["cls"][indices])
    for job in references:
        crop = args.reference / job["crop"]
        if sha(crop) != job["crop_sha256"]:
            raise ValueError("reference crop identity mismatch")
        shutil.copyfile(crop, root / "reference/crops" / crop.name)
        job["crop"] = f"reference/crops/{crop.name}"
    write(root / "reference/jobs.json", references)
    shutil.copyfile(label_path, root / "reference/labels.json")
    write(root / "jobs.json", jobs)
    write(root / "candidates-prepared.json", prepared)
    protocol = {
        "schema": "parts_full_grasp_classification_v2", "dataset_id": root.name,
        "source_dataset": str(args.source), "source_candidates_sha256": sha(args.source / "candidates.json"),
        "reference_dataset": str(args.reference), "reference_source_features_sha256": sha(feature_path),
        "reference_labels_sha256": sha(label_path), "reference_labels_user_approved": False,
        "candidates": len(prepared), "eligible_images": len(jobs),
        "candidates_with_lifted_closed_images": sum(bool(c["eligible_frame_ids"]) for c in prepared),
        "per_episode_images": dict(Counter(j["episode_number"] for j in jobs)),
        "selection": "post-close, within candidate, valid FK, openness<0.60, entry-relative base-Z>=0.05m, capture age 0..0.25s",
        "height_is_sufficient_for_success": False,
        "classification": "Frozen SAM3 adaptive finger points; frozen DINOv3 ROI/CLS nearest reference cosine margins +/-0.02; disagreement -> unknown",
        "reference_exclusion": "exclude the whole query episode and matching image SHA from both feature banks",
        "temporal_proposal": "two agreeing fresh distinct samples spanning >=0.12s with gap<=0.25s; all lifted snapshots empty for empty proposal; mixed held/empty -> review",
        "temporal_thresholds_are_offline_proposals": True, "tracking": False,
        "training_executed": False, "ready_for_training": False, "approved_rewards": 0,
        "pipeline_sha256": sha(Path(__file__)),
    }
    write(root / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == "__main__":
    main()
