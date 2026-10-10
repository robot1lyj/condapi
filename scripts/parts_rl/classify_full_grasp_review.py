"""Build pooled crops and classify all grasp proposals without reward approval."""
# ruff: noqa: RUF001

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
from PIL import ImageDraw
from prepare_full_grasp_review import sha
from prepare_full_grasp_review import write


def prepare_crops(root):
    output = root / "crops"
    output.mkdir(exist_ok=False)
    measured = {r["job"]["id"]: r for r in json.loads((root / "roi-results.json").read_text())["records"]}
    records, skipped = [], []
    for job in json.loads((root / "jobs.json").read_text()):
        measurement = measured[job["id"]]["measurement"]
        if not measurement["geometry_valid"]:
            skipped.append({"id": job["id"], "reason": measurement.get("reason", "invalid fingertip geometry")})
            continue
        source = root / job["image"]
        if sha(source) != job["sha256"]:
            raise ValueError("source image identity mismatch")
        image = Image.open(source).convert("RGB")
        center = np.mean(measurement["tips_xy"], axis=0)
        left, top = round(center[0]) - 128, round(center[1]) - 152
        box = (left, top, left + 256, top + 256)
        if left < 0 or top < 0 or box[2] > image.width or box[3] > image.height:
            skipped.append({"id": job["id"], "reason": "fixed crop outside source; no padding"})
            continue
        with np.load(root / "roi" / (job["id"] + ".npz")) as data:
            roi = data["roi"][top : top + 256, left : left + 256]
        weights = roi.reshape(16, 16, 16, 16).mean(axis=(1, 3)).astype(np.float32)
        if weights.sum() <= 0:
            raise ValueError("valid geometry unexpectedly has no patch support")
        crop_path = output / (job["id"] + ".png")
        image.crop(box).save(crop_path)
        weights_path = output / (job["id"] + "-weights.npz")
        np.savez_compressed(weights_path, weights=weights)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box, outline="cyan", width=2)
        draw.polygon([tuple(p) for p in measurement["polygon_xy"]], outline="yellow", width=2)
        overlay.save(output / (job["id"] + "-source.jpg"), quality=90)
        records.append({**job, "crop": str(crop_path.relative_to(root)), "crop_sha256": sha(crop_path),
                        "patch_weights": str(weights_path.relative_to(root)), "crop_box_xyxy": box,
                        "roi_measurement": measurement})
    write(root / "feature-jobs.json", records)
    write(root / "crop-run.json", {"images": len(records), "skipped": skipped, "automatic_tracking": False,
                                   "pipeline_sha256": sha(Path(__file__))})
    print(json.dumps({"prepared_crops": len(records), "skipped": len(skipped)}))


def normalized(features):
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("invalid feature matrix")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("zero feature")
    return features / norms


def snapshot_prediction(query, vector, references, reference_features, labels):
    """Exclude all same-episode exemplars, including duplicate images under new IDs."""
    allowed = [i for i, j in enumerate(references) if j["episode_number"] != query["episode_number"]
               and j["sha256"] != query["sha256"] and j["roi_measurement"]["geometry_valid"]]
    nearest = {}
    for label in ["held_at_snapshot", "empty_at_snapshot"]:
        group = [i for i in allowed if labels[references[i]["id"]] == label]
        if not group:
            return {"label": "uncertain", "margin": None, "nearest": {}, "reason": "missing independent reference class"}
        scores = reference_features[group] @ vector
        index = int(np.argmax(scores))
        chosen = references[group[index]]
        nearest[label] = {"id": chosen["id"], "episode_number": chosen["episode_number"],
                          "similarity": float(scores[index]), "crop": chosen["crop"]}
    margin = nearest["held_at_snapshot"]["similarity"] - nearest["empty_at_snapshot"]["similarity"]
    label = "held_at_snapshot" if margin >= .02 else "empty_at_snapshot" if margin <= -.02 else "uncertain"
    return {"label": label, "margin": margin, "nearest": nearest}


def evidence_span(frames, target):
    """Count only advancing, distinct camera observations within a short contiguous run."""
    best, start, previous = 0., None, None
    for frame in frames:
        connected = previous is not None and frame["epoch"] == previous["epoch"]
        connected = connected and 0 < frame["visual_time"] - previous["visual_time"] <= .25
        connected = connected and frame["camera_sequence"] > previous["camera_sequence"]
        connected = connected and frame["sha256"] != previous["sha256"]
        if frame["label"] != target:
            start = None
        else:
            start = start if start is not None and connected else frame["visual_time"]
            best = max(best, frame["visual_time"] - start)
        previous = frame
    return float(best)


def aggregate(frames):
    held = evidence_span(frames, "held_at_snapshot")
    empty = evidence_span(frames, "empty_at_snapshot")
    labels = {f["label"] for f in frames}
    if "held_at_snapshot" in labels and "empty_at_snapshot" in labels:
        return "uncertain", held, empty, "mixed_held_and_empty"
    if held >= .12:
        return "held_observed", held, empty, "short_multiframe_held_evidence"
    if frames and labels == {"empty_at_snapshot"} and empty >= .12:
        return "empty_observed", held, empty, "all_lifted_snapshots_empty"
    return "uncertain", held, empty, "no_eligible_snapshot" if not frames else "insufficient_or_conflicting_evidence"


def finalize(root):
    jobs = json.loads((root / "jobs.json").read_text())
    feature_jobs_path = root / "feature-jobs.json"
    feature_jobs = json.loads(feature_jobs_path.read_text())
    run = json.loads((root / "dinov3/run.json").read_text())
    if run["jobs_sha256"] != sha(feature_jobs_path):
        raise ValueError("feature job identity mismatch")
    references = json.loads((root / "reference/jobs.json").read_text())
    label_path = root / "reference/labels.json"
    protocol = json.loads((root / "protocol.json").read_text())
    if sha(label_path) != protocol["reference_labels_sha256"]:
        raise ValueError("reference labels changed")
    labels = {r["candidate_id"]: r["label"] for r in json.loads(label_path.read_text())["records"]}
    if set(labels) != {j["id"] for j in references}:
        raise ValueError("reference labels must match bank exactly")
    with np.load(root / "reference/features.npz") as ref, np.load(root / "dinov3/features.npz") as query:
        if ref["ids"].tolist() != [j["id"] for j in references] or query["ids"].tolist() != [j["id"] for j in feature_jobs]:
            raise ValueError("feature order mismatch")
        banks = {name: normalized(ref[name]) for name in ["roi", "cls"]}
        features = {name: normalized(query[name]) for name in ["roi", "cls"]}
    measured = {r["job"]["id"]: r for r in json.loads((root / "roi-results.json").read_text())["records"]}
    skipped = {r["id"]: r["reason"] for r in json.loads((root / "crop-run.json").read_text())["skipped"]}
    feature_indices = {j["id"]: i for i, j in enumerate(feature_jobs)}
    frames = {}
    for job in jobs:
        cid = job["id"]
        frame = {**job, "label": "uncertain", "roi": None, "cls": None,
                 "measurement": measured[cid]["measurement"], "overlay": measured[cid]["visualization"],
                 "reference_exclusion": "same_episode_and_matching_sha", "approved_reward": None}
        if cid in skipped:
            frame["reason"] = skipped[cid]
        else:
            i = feature_indices[cid]
            for name in ["roi", "cls"]:
                frame[name] = snapshot_prediction(job, features[name][i], references, banks[name], labels)
            a, b = frame["roi"]["label"], frame["cls"]["label"]
            frame["label"] = a if a == b else "uncertain"
            frame["reason"] = "feature_agreement" if a == b and a != "uncertain" else "feature_ambiguous_or_disagree"
        frames[cid] = frame
    prepared = json.loads((root / "candidates-prepared.json").read_text())
    context_dir = root / "context"
    context_dir.mkdir(exist_ok=False)
    source_root = Path(protocol["source_dataset"])
    source_jobs = {j["frame_key"]: j for j in json.loads((source_root / "image-jobs.json").read_text())}
    candidates, cards = [], []
    reason_text = {
        "short_multiframe_held_evidence": "多张新鲜抬升画面均支持持物；这里只提出持物证据，不批准成功奖励。",
        "all_lifted_snapshots_empty": "可用抬升画面均支持空手；可能存在更早的抓住后放开，仍需核对完整回放。",
        "mixed_held_and_empty": "同一次候选出现持物与空手证据；可能掉落、提前松开或视觉误判，保留待复核。",
        "no_eligible_snapshot": "没有满足闭爪、抬升及图像新鲜条件的画面，不能把缺少证据判成失败。",
        "insufficient_or_conflicting_evidence": "可靠多帧证据不足、夹指几何无效或DINO两分支冲突，保留待复核。",
    }
    for candidate in prepared:
        evidence = [frames[key] for key in candidate["eligible_frame_ids"]]
        label, held_s, empty_s, reason = aggregate(evidence)
        gallery = [{"raw": frame["image"], "overlay": frame["overlay"], "row": frame["row"],
                            "delta_m": frame["lift_delta_m"], "held": frame["label"] == "held_at_snapshot",
                            "label": frame["label"], "reason": frame["reason"],
                            "roi_margin": frame["roi"]["margin"] if frame["roi"] else None,
                            "cls_margin": frame["cls"]["margin"] if frame["cls"] else None,
                            "nearest": frame["roi"]["nearest"] if frame["roi"] else {}} for frame in evidence]
        reference_gallery = []
        for key, caption in [("entry_row", "height reference / before closure"), ("close_row", "closure threshold")]:
            sample = next(s for s in candidate["samples"] if s["row"] == candidate[key])
            source_job = source_jobs[sample["frame_key"]]
            source_image = source_root / source_job["image"]
            if sha(source_image) != source_job["sha256"]:
                raise ValueError("context image identity mismatch")
            destination = context_dir / source_image.name
            shutil.copyfile(source_image, destination)
            reference_gallery.append({"image": str(destination.relative_to(root)), "row": sample["row"],
                                      "caption": caption, "base_z_m": sample["base_z_m"],
                                      "openness": sample["openness"], "sha256": source_job["sha256"]})
        if not gallery:
            fallback = next(s for s in candidate["samples"] if s["image"] == candidate["fallback_image"])
            gallery.append({"raw": fallback["image"], "overlay": fallback["image"], "row": fallback["row"],
                            "delta_m": fallback["lift_delta_m"], "held": False, "label": "not_eligible",
                            "reason": "no_eligible_snapshot", "roi_margin": None, "cls_margin": None, "nearest": {}})
        result = {**candidate, "suggested_label": label, "classification_reason": reason,
                  "visual_hold_s": held_s, "clear_empty_s": empty_s, "review_gallery": gallery,
                  "reference_gallery": reference_gallery,
                  "frame_label_counts": dict(Counter(f["label"] for f in evidence)),
                  "reasons": [reason_text[reason], "高度为各臂base-Z相对候选进入点，未作桌面标定；5cm不是抓住的充分条件。",
                              "同集参考整集排除；快照持物不等于正确颜色放置，不启用自动跟踪。"],
                  "review_label": None, "approved_reward": None, "review_status": "unreviewed"}
        result["proposed_reward"] = {"held_observed": 1., "empty_observed": 0., "uncertain": None}[label]
        if candidate["end_reason"] != "reopened":
            result["reasons"].append("片段在录像结束、数据断开或候选窗口上限处终止，须检查是否完整。")
        candidates.append(result)
        cards.append({k: v for k, v in result.items() if k != "samples"})
    counts = dict(Counter(c["suggested_label"] for c in candidates))
    summary = {"schema": "parts_full_grasp_review_v2", "dataset_id": root.name, "episodes": 14,
               "candidates": len(candidates), "images": len(jobs), "dinov3_images": len(feature_jobs),
               "suggested_counts": counts, "frame_label_counts": dict(Counter(f["label"] for f in frames.values())),
               "reason_counts": dict(Counter(c["classification_reason"] for c in candidates)),
               "per_episode": {str(ep): dict(Counter(c["suggested_label"] for c in candidates if c["episode_number"] == ep))
                               for ep in range(1, 15)},
               "per_arm": {arm: dict(Counter(c["suggested_label"] for c in candidates if c["arm"] == arm))
                           for arm in ["left", "right"]},
               "source_candidates_sha256": protocol["source_candidates_sha256"],
               "all_candidates_retained": len(candidates) == protocol["candidates"],
               "reference_exclusion": protocol["reference_exclusion"], "reference_images": len(references),
               "reference_labels_user_approved": False, "independent_accuracy_measured": False,
               "automatic_tracking": False, "training_executed": False, "approved_rewards": 0,
               "reviewed": 0, "ready_for_training": False, "automatic_reward_ready": False,
               "hold_evidence_min_span_s": .12, "max_capture_gap_s": .25,
               "protocol_sha256": sha(root / "protocol.json"), "pipeline_sha256": sha(Path(__file__))}
    # Reproducible balanced spot-check list, drawn without using outcomes or operator labels.
    rng = random.Random(20261010)
    spot = []
    for label in ["held_observed", "empty_observed", "uncertain"]:
        for arm in ["left", "right"]:
            pool = [c for c in candidates if c["suggested_label"] == label and c["arm"] == arm]
            spot.extend(c["candidate_id"] for c in rng.sample(pool, min(4, len(pool))))
    for card in cards:
        card["spot_check"] = card["candidate_id"] in spot
    summary["spot_check_candidates"] = len(spot)
    write(root / "spot-check.json", {"seed": 20261010, "stratification": "suggestion x arm; up to 4 each",
                                    "candidate_ids": spot, "review_label": None})
    write(root / "frame-classifications.json", list(frames.values()))
    write(root / "candidates.json", candidates)
    write(root / "summary.json", summary)
    columns = ["candidate_id", "episode_number", "arm", "suggested_label", "proposed_reward", "classification_reason",
               "visual_hold_s", "clear_empty_s", "review_label", "review_notes", "approved_reward", "clip"]
    with (root / "review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for c in candidates:
            writer.writerow({k: c.get(k, "") for k in columns} | {"clip": c["review_clip"]["path"]})
    template = Path(__file__).with_name("full_review_template.html").read_text()
    data = json.dumps({"summary": summary, "cards": cards}, ensure_ascii=False).replace("<", "\\u003c")
    with (root / "index.html").open("x") as stream:
        stream.write(template.replace("__REVIEW_DATA__", data))
    (root / "README.txt").write_text(
        "打开index.html，按分类/集数/机械臂筛选并回放。勾选抽检清单查看固定随机抽样；可审核、暂存、导出JSON。\n"
        "三类为持物证据、空手证据、待复核，均不是已批准的奖励。原始273个候选全部保留。\n"
        "原图和分割叠图点击切换；多帧分类只比较外观，不追踪物体。相似度差不是概率。\n"
        "第一集未用于RL训练，但已参与感知调试，不是全流程未见验证集。没有RL训练或生产接入。\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage", choices=["crops", "classify"], required=True)
    args = parser.parse_args()
    (prepare_crops if args.stage == "crops" else finalize)(args.dataset)


if __name__ == "__main__":
    main()
