"""Create a portable offline HTML/JSON/CSV grasp-review dataset. No training."""
# ruff: noqa: RUF001 -- Human-facing Chinese punctuation is intentional.

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/parts-rl/src"))
from parts_rl.candidates import suggest_outcome


def save_json(path, value):
    if path.exists():
        raise ValueError(f"write a new review artifact: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def overlay(root, candidate, sample, prediction):
    image = Image.open(root / sample["image"]).convert("RGBA")
    with np.load(root / "vision" / (sample["frame_key"] + ".npz"), allow_pickle=False) as archive:
        h, w = archive["shape"]
        masks = np.unpackbits(archive["masks_packed"], axis=1)[:, :h * w].reshape(-1, h, w)
    for name, color in (("bin", (40, 120, 230)), ("brick", (30, 215, 100))):
        for obj in prediction["detections"][name]:
            alpha = Image.fromarray(masks[obj["mask_index"]].astype(np.uint8) * 70)
            tint = Image.new("RGBA", image.size, (*color, 0))
            tint.putalpha(alpha)
            image = Image.alpha_composite(image, tint)
    draw = ImageDraw.Draw(image)
    for obj in prediction["detections"]["brick"]:
        if obj["score"] >= 0.5:
            draw.rectangle(obj["box"], outline="#22dd66", width=2)
    draw.rectangle((0, 0, image.width, 25), fill="#111827")
    delta = sample["lift_delta_m"]
    text = f"{candidate['candidate_id']} row={sample['row']} dz={delta:.3f}m" if delta is not None else "unknown pose"
    draw.text((6, 7), text, fill="white")
    name = f"overlays/{candidate['candidate_id']}-row{sample['row']}.jpg"
    image.convert("RGB").save(root / name, quality=90)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    raw = json.loads((root / "candidates-unscored.json").read_text())
    jobs = json.loads((root / "image-jobs.json").read_text())
    predictions = {}
    for job in jobs:
        path = root / "vision" / (job["frame_key"] + ".json")
        if not path.is_file():
            raise ValueError(f"incomplete inference: {job['frame_key']}")
        prediction = json.loads(path.read_text())
        if prediction["image_sha256"] != job["sha256"]:
            raise ValueError("prediction/source identity mismatch")
        predictions[job["frame_key"]] = prediction
    clips = {x["candidate_id"]: x for x in json.loads((root / "clip-manifest.json").read_text())}
    (root / "overlays").mkdir(exist_ok=False)
    candidates, cards = [], []
    for candidate in raw:
        entry = next(s for s in candidate["samples"] if s["row"] == candidate["entry_row"])
        if candidate["reference_row"] != candidate["entry_row"]:
            candidate["earlier_provisional_reference"] = {
                "row": candidate["reference_row"], "base_z_m": candidate["reference_base_z_m"],
                "method": "minimum_near_close_not_used_for_final_suggestion"}
        candidate.update(reference_row=candidate["entry_row"], reference_base_z_m=entry["base_z_m"],
                         reference_method="candidate_entry_base_z")
        result = suggest_outcome(candidate, predictions)
        result["review_clip"] = clips[result["candidate_id"]]
        selected = {0, len(result["samples"]) - 1}
        selected.add(max(range(len(result["samples"])), key=lambda i: bool(result["samples"][i]["held_lifted"])))
        selected.add(max(range(len(result["samples"])), key=lambda i: (
            result["samples"][i]["lift_delta_m"] if result["samples"][i]["lift_delta_m"] is not None else -np.inf)))
        gallery = []
        for index in sorted(selected):
            sample = result["samples"][index]
            picture = overlay(root, result, sample, predictions[sample["frame_key"]])
            gallery.append({"raw": sample["image"], "overlay": picture, "row": sample["row"],
                            "delta_m": sample["lift_delta_m"], "held": sample["held_lifted"]})
        result["review_gallery"] = gallery
        candidates.append(result)
        cards.append({key: result[key] for key in ("candidate_id", "episode_number", "arm", "split", "suggested_label",
                                                  "entry_row", "close_row", "end_row", "reasons", "max_lift_delta_m",
                                                  "visual_hold_s", "clear_empty_s", "review_clip", "review_gallery")})
    counter = Counter(c["suggested_label"] for c in candidates)
    counts = {label: counter[label] for label in ("success", "failure", "uncertain")}
    geometry_counts = {name: sum(bool(p["geometry"].get(name)) for p in predictions.values())
                       for name in ("holding_candidate", "empty_candidate", "gripper_seen")}
    summary = {"schema": "parts_grasp_review_dataset_v1", "dataset_id": root.name, "episodes": 14,
               "candidates": len(candidates), "images": len(jobs), "suggested_counts": counts,
               "per_episode": {str(n): dict(Counter(c["suggested_label"] for c in candidates if c["episode_number"] == n))
                               for n in range(1, 15)}, "reviewed": 0, "approved_rewards": 0,
               "ready_for_training": False, "training_executed": False,
               "up_axis_status": "provisional_base_z_requires_review", "lift_m": 0.05, "hold_s": 1.0,
               "all_candidates_retained": True, "vision_pipeline_shas": sorted({p["pipeline_sha256"] for p in predictions.values()}),
               "reference_method": "candidate_entry_base_z", "vision_geometry_counts": geometry_counts,
               "automatic_reward_ready": False,
               "checkpoint_shas": sorted({p["checkpoint_sha256"] for p in predictions.values()})}
    save_json(root / "candidates.json", candidates)
    save_json(root / "summary.json", summary)
    columns = ["candidate_id", "episode_number", "arm", "split", "suggested_label", "review_label",
               "entry_row", "close_row", "end_row", "max_lift_delta_m", "visual_hold_s", "clear_empty_s", "clip", "reason", "review_notes"]
    with (root / "review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({key: candidate.get(key, "") for key in columns} | {
                "clip": candidate["review_clip"]["path"], "reason": "；".join(candidate["reasons"])})
    template = Path(__file__).with_name("review_template.html").read_text()
    data = json.dumps({"summary": summary, "cards": cards}, ensure_ascii=False).replace("<", "\\u003c")
    (root / "index.html").write_text(template.replace("__REVIEW_DATA__", data))
    (root / "README.txt").write_text(
        "打开 index.html 逐条播放、选择审核标签并导出审核JSON。页面可离线使用。\n"
        "自动成功/失败均为候选建议，全部尚未审核；approved_reward全部为空，未开始RL训练。\n"
        "不确定项、重复尝试、可能的非抓取调整全部保留。第一集独立留验。\n"
        "注意：相对抬升暂用各臂base-Z、候选进入行参考；需审核参考点与向上轴。\n"
        "成功候选需持物几何与新鲜视觉连续1秒、相对抬升5cm；零检测不直接标失败。\n"
        "CSV也可填写，但建议导出JSON保留数据集身份和逐条审核状态。\n")
    payload = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in sorted(root.rglob("*")) if path.is_file()]
    save_json(root / "payload-manifest.json", payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
