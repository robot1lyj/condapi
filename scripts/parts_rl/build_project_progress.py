"""Build a portable project report from immutable candidate and qualification data.

CPU media preparation only; no neural inference, training or reward publication.
"""

# ruff: noqa: RUF001

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagram(path, prefix):
    svg = re.search(r"<svg\b[\s\S]*?</svg>", path.read_text()).group(0)
    # Isolate IDs/markers when several independently validated SVGs share a page.
    ids = re.findall(r'\bid="([^"]+)"', svg)
    for old in sorted(ids, key=len, reverse=True):
        svg = svg.replace(f'id="{old}"', f'id="{prefix}-{old}"')
        svg = svg.replace(f"url(#{old})", f"url(#{prefix}-{old})")
    svg = re.sub(
        r'aria-labelledby="([^"]+)"',
        lambda m: 'aria-labelledby="' + " ".join(prefix + "-" + x for x in m[1].split()) + '"',
        svg,
    )
    # Inline overview is static; the full linked viewer supplies node interaction.
    return re.sub(r' tabindex="[^"]*"| role="button"| aria-pressed="[^"]*"', "", svg)


def model_execution(dataset, output, candidate, asset_hashes):
    """Show saved model inputs/outputs; boxes describe actual fixed preprocessing."""
    jobs = {j["id"]: j for j in read(dataset / "feature-jobs.json")}
    frame = candidate["frames"][-1]
    key = frame["views"][candidate["arm"]]["key"]
    query = jobs[key]
    scores = read(dataset / "frame-classifications.json")[key]

    def copy_asset(source, name, expected=None):
        if expected and digest(source) != expected:
            raise ValueError("Model example input identity mismatch")
        path = "progress-media/" + name
        shutil.copyfile(source, output / path)
        asset_hashes[path] = digest(output / path)
        return path

    original = copy_asset(
        dataset / frame["views"][candidate["arm"]]["image"],
        "dino-query-original.jpg",
        query["source_sha256"],
    )
    with Image.open(output / original) as image:
        if image.size != (640, 480):
            raise ValueError("Original image size does not match the box illustration")
    crop = copy_asset(dataset / query["crop"], "dino-query-crop.png", query["crop_sha256"])
    with np.load(dataset / query["patch_weights"]) as saved:
        weights = saved["weights"]
    expected = np.zeros((16, 16), np.float32)
    expected[3:16, 4:12] = 1
    if not np.array_equal(weights, expected):
        raise ValueError("Example ROI must match the executed patch weights")
    references = []
    for label in ["held_at_snapshot", "empty_at_snapshot"]:
        nearest = scores["roi"]["nearest"][label]
        job = jobs[nearest["id"]]
        if job["episode_number"] == query["episode_number"] or job["source_sha256"] == query["source_sha256"]:
            raise ValueError("Leaked reference in model example")
        references.append(
            {
                "id": job["id"],
                "label": label,
                "image": copy_asset(dataset / job["crop"], f"dino-{label}.png", job["crop_sha256"]),
                "similarity": nearest["similarity"],
            }
        )
    raw = candidate["vlm"]
    images = []
    for i, row in enumerate(raw["evidence_image_rows"]):
        source = next(f for f in candidate["frames"] if f["row"] == row)
        name = f"progress-media/qwen-input-{chr(65 + i)}.png"
        with Image.open(dataset / source["views"][candidate["arm"]]["image"]) as im:
            image = im.convert("RGB")
            if i:
                image = image.crop((140, 140, 500, 480))
            image.save(output / name)
        asset_hashes[name] = digest(output / name)
        images.append(
            {
                "image": name,
                "letter": chr(65 + i),
                "row": row,
                "time_s": source["since_close_s"],
                "cited": chr(65 + i) in raw["decision"]["evidence_frames"],
            }
        )
    return {
        "candidate_id": candidate["candidate_id"],
        "row": frame["row"],
        "original": original,
        "crop": crop,
        "fixed_crop_xyxy": [160, 180, 480, 480],
        "roi_patch_rows": [3, 16],
        "roi_patch_cols": [4, 12],
        "roi_box_in_resized_crop": [64, 48, 192, 256],
        "scores": scores,
        "roi_references": references,
        "dino_attempt_outcome": candidate["dino"]["outcome"],
        "qwen_inputs": images,
        "qwen_raw_response": raw["raw_response"],
        "qwen_model": raw["model_id"],
        "qwen_prompt_sha256": raw["prompt_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    source = read(args.dataset / "candidates.json")
    summary = read(args.qualification / "summary.json")
    if digest(args.dataset / "candidates.json") != summary["source_candidates_sha256"]:
        raise ValueError("Stale qualification identity")
    rows = read(args.qualification / "lift-qualified-proposals.json")
    if len(rows) != len(source) or {r["candidate_id"] for r in rows} != {r["candidate_id"] for r in source}:
        raise ValueError("Candidate/qualification IDs differ")
    candidates = {c["candidate_id"]: c for c in source}
    qualified = {c["candidate_id"]: c for c in rows}
    clips = {c["candidate_id"]: c for c in read(args.dataset / "clip-manifest.json")}
    eligible = [r for r in rows if r["evidence_eligible"]]
    labels = Counter(r["effective_label"] for r in eligible)
    reviewed = [r for r in rows if r["review_label"]]
    total = len(rows)
    data = {
        "episodes": len({r["episode_number"] for r in rows}),
        "candidates": total,
        "eligible": len(eligible),
        "excluded": total - len(eligible),
        "success": labels["success"],
        "failure": labels["failure"],
        "classified": sum(labels.values()),
        "classification_completion_percent": round(sum(labels.values()) / len(eligible) * 100, 1),
        "qualification_percent": round(len(eligible) / total * 100, 1),
        "grasp_success_share_percent": round(labels["success"] / len(eligible) * 100, 1),
        "user_reviewed": len(reviewed),
        "original_proposals_agree": sum(r["review_label"] == r["source_label"] for r in reviewed),
        "review_corrections": sum(r["review_label"] != r["source_label"] for r in reviewed),
        "reviewed_eligible": sum(r["evidence_eligible"] for r in reviewed),
        "medium_total": sum(c["confidence"] == "medium" for c in source),
        "medium_reviewed": sum(candidates[r["candidate_id"]]["confidence"] == "medium" for r in reviewed),
        "approved_rewards": sum(r["approved_reward"] is not None for r in rows),
        "single_lift_image_count": len(summary["single_selected_lift_frame_ids"]),
        "excluded_ids": summary["excluded_ids"],
        "source_candidates_sha256": summary["source_candidates_sha256"],
        "human_review_sha256": summary["human_review_sha256"],
        "qualification_sha256": digest(args.qualification / "lift-qualified-proposals.json"),
        "training_executed": False,
    }
    output = args.output_directory
    media = output / "progress-media"
    media.mkdir(exist_ok=True)
    selection = [
        ("E01-L004-v3", "success", "抓住并持物", "抬升后前景积木仍在指间；用户抽检确认为抓到。"),
        ("E05-L006-v3", "failure", "抬升后的空抓", "已有抬升证据，核验画面指间为空；两模型与用户结论一致。"),
        (
            "E02-R007-v3",
            "correction",
            "纠正背景造成的误报",
            "Qwen最初报成功，DINO证据混合；序列复核和用户抽检确认未持住。",
        ),
        (
            "E04-L015-v3",
            "excluded",
            "没有抬升，不记空抓0",
            "闭爪后仍下降，核验窗内没有抬升；用户改为非独立抓取，候选奖励为null。",
        ),
        (
            "E04-L016-v3",
            "success",
            "下一次重试才真正抬升",
            "该次候选有自己的抬升和持物证据；不能把这段的成功借给上一段。",
        ),
    ]
    samples, asset_hashes = [], {}
    traces = {}
    for cid, category, title, explanation in selection:
        c, q = candidates[cid], qualified[cid]
        frames = c["frames"]
        close = min(frames, key=lambda f: abs(f["row"] - c["close_row"]))
        post = [f for f in frames if f["row"] >= c["close_row"]]
        lifted = [f for f in post if f["lift_from_closure_m"] >= 0.05]
        middle = lifted[0] if lifted else post[len(post) // 2]
        chosen = [close, middle, post[-1]]
        images = []
        for k, f in enumerate(chosen):
            name = f"progress-media/{cid}-{f['row']}.jpg"
            shutil.copyfile(args.dataset / f["views"][c["arm"]]["image"], output / name)
            crop = f"progress-media/{cid}-{f['row']}-crop.jpg"
            with Image.open(output / name) as image:
                image.convert("RGB").crop((160, 180, 480, 480)).save(output / crop, quality=95)
            for asset in [name, crop]:
                asset_hashes[asset] = digest(output / asset)
            images.append(
                {
                    "image": name,
                    "crop": crop,
                    "row": f["row"],
                    "time_s": round(f["since_close_s"], 3),
                    "rise_cm": round(f["lift_from_closure_m"] * 100, 2),
                    "stage": "闭爪附近"
                    if k == 0
                    else "首次合格抬升图"
                    if k == 1 and lifted
                    else "核验中段"
                    if k == 1
                    else "释放前核验末图",
                }
            )
        clip = f"progress-media/{cid}.mp4"
        if digest(args.dataset / clips[cid]["path"]) != clips[cid]["sha256"]:
            raise ValueError("Clip hash mismatch")
        shutil.copyfile(args.dataset / clips[cid]["path"], output / clip)
        asset_hashes[clip] = digest(output / clip)
        ep = c["episode_number"]
        if ep not in traces:
            traces[ep] = read(args.dataset / "traces" / f"E{ep:02d}.json")
        trace = traces[ep]
        curve = [
            [
                round(trace["time"][r] - trace["time"][c["close_row"]], 4),
                round(
                    (trace["arms"][c["arm"]]["position_m"][r][2] - q["lift_evidence"]["reference_base_z_m"]) * 100, 4
                ),
            ]
            for r in range(c["close_row"], c["inspection_end_row"] + 1)
        ]
        samples.append(
            {
                "id": cid,
                "category": category,
                "title": title,
                "explanation": explanation,
                "arm": c["arm"],
                "images": images,
                "clip": clip,
                "curve": curve,
                "max_rise_cm": round(q["lift_evidence"]["max_postclosure_rise_m"] * 100, 2),
                "vlm": c["vlm"]["decision"]["outcome"],
                "dino": c["dino"]["outcome"],
                "effective_label": q["effective_label"],
                "review_label": q["review_label"],
                "eligible": q["evidence_eligible"],
                "proposed_reward": q["proposed_reward"],
            }
        )
    data["samples"] = samples
    data["execution"] = model_execution(args.dataset, output, candidates["E01-L004-v3"], asset_hashes)
    data["assets_sha256"] = asset_hashes
    template = Path(__file__).with_name("project_progress_template.html").read_text()
    for name, prefix in [
        ("grasp-judgement-architecture", "va"),
        ("qwen-dino-detail", "qd"),
        ("pld-rl-architecture", "rl"),
    ]:
        template = template.replace("__" + prefix.upper() + "_SVG__", diagram(output / (name + ".html"), prefix))
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    (output / "project-progress.html").write_text(template.replace("__DATA__", encoded))
    (output / "project-progress-data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output / "project-progress.html"),
                "candidates": total,
                "eligible": len(eligible),
                "media_files": len(asset_hashes),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
