"""Inspect a fixed fingertip-bridge color rule; no tracking or reward approval."""
# ruff: noqa: RUF001

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw


def color_pixels(rgb, *, shadow_tolerant=False):
    values = rgb.astype(np.int16)
    high = values.max(2)
    chroma = high - values.min(2)
    if shadow_tolerant:
        return (high >= 30) & (chroma >= 12) & (chroma >= high * 0.20)
    return (high >= 60) & (chroma >= 45) & (chroma >= high * 0.25)


def finger_tip(mask, anchor):
    """Use the seeded component and its robust upper end, not the prompt point."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    x, y = map(int, anchor)
    component = int(labels[y, x])
    if component == 0 or count < 2 or stats[component, cv2.CC_STAT_AREA] < 300:
        raise ValueError("finger seed has no sufficiently large component")
    selected = labels == component
    ys, xs = np.nonzero(selected)
    tip_y = int(np.quantile(ys, 0.01))
    tip_x = int(np.median(xs[(ys >= tip_y) & (ys <= tip_y + 4)]))
    return [tip_x, tip_y], selected


def bridge_mask(shape, left, right, half_width=6.0):
    """A 12px-wide oriented band connecting the two inferred tips."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    vector = b - a
    distance = float(np.linalg.norm(vector))
    if distance < 8 or distance > 300 or a[0] >= b[0]:
        raise ValueError("invalid or tiny fingertip gap")
    normal = np.array([-vector[1], vector[0]]) / distance * half_width
    polygon = np.rint([a + normal, b + normal, b - normal, a - normal]).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool), polygon.tolist(), distance


def measure(rgb, predictions, root):
    masks, tips, quality, darkness = [], [], [], []
    for item in predictions:
        with np.load(root / item["masks"]) as saved:
            h, w = saved["shape"]
            unpacked = np.unpackbits(saved["masks_packed"], axis=1)[:, : h * w].reshape(-1, h, w)
        index = item["selected_index"]
        tip, component = finger_tip(unpacked[index], item["points_xy"][0])
        masks.append(component)
        tips.append(tip)
        quality.append(float(item["predicted_mask_quality"][index]))
        values = rgb.astype(np.int16)
        dark = (values.max(2) < 110) & (values.max(2) - values.min(2) < 60)
        darkness.append(float(dark[component].mean()))
    full, polygon, gap = bridge_mask(rgb.shape[:2], *tips)
    roi = full & ~masks[0] & ~masks[1]
    area = int(roi.sum())
    usable_fraction = area / int(full.sum())
    colored = color_pixels(rgb) & roi
    fraction = float(colored.sum() / area) if area else None
    shadow_fraction = float(color_pixels(rgb, shadow_tolerant=True)[roi].mean()) if area else None
    valid = area >= 48 and usable_fraction >= 0.5 and min(quality) >= 0.5 and min(darkness) >= 0.6
    result = {
        "tips_xy": tips,
        "polygon_xy": polygon,
        "gap_px": gap,
        "roi_pixels": area,
        "usable_fraction": usable_fraction,
        "color_pixels": int(colored.sum()),
        "color_fraction": fraction,
        "finger_quality": quality,
        "finger_dark_fraction": darkness,
        "shadow_tolerant_color_fraction": shadow_fraction,
        "geometry_valid": bool(valid),
        "threshold_met": bool(fraction > 0.8) if valid else None,
        "shadow_tolerant_threshold_met": bool(shadow_fraction > 0.8) if valid else None,
        "approved_reward": None,
    }
    return result, roi, colored, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, help="Optional qualitative snapshot audit; never approves rewards")
    args = parser.parse_args()
    root = args.dataset
    output = root / "roi"
    output.mkdir(exist_ok=False)
    records = []
    audit = json.loads(args.audit_json.read_text()) if args.audit_json else None
    audit_by_id = {item["candidate_id"]: item for item in audit["records"]} if audit else {}
    for job in json.loads((root / "jobs.json").read_text()):
        image_path = root / job["image"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != job["sha256"]:
            raise ValueError("image identity mismatch")
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        pred = json.loads((root / "point-adaptive-predictions" / (job["id"] + ".json")).read_text())
        if pred["job"]["sha256"] != job["sha256"]:
            raise ValueError("prediction identity mismatch")
        overlay = rgb.copy()
        try:
            if len(pred["results"]) != 2:
                raise ValueError(pred.get("unknown_reason", "two finger segmentations required"))
            result, roi, colored, masks = measure(rgb, pred["results"], root)
            for mask in masks:
                overlay[mask] = (overlay[mask] * 0.65 + np.array([30, 170, 90]) * 0.35).astype(np.uint8)
            overlay[roi] = (overlay[roi] * 0.5 + np.array([250, 40, 230]) * 0.5).astype(np.uint8)
            draw_image = Image.fromarray(overlay)
            draw = ImageDraw.Draw(draw_image)
            draw.polygon([tuple(x) for x in result["polygon_xy"]], outline="yellow", width=2)
            for x, y in result["tips_xy"]:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="cyan")
            overlay = np.asarray(draw_image)
            np.savez_compressed(output / (job["id"] + ".npz"), roi=roi, colored=colored)
        except ValueError as exc:
            result = {
                "geometry_valid": False,
                "threshold_met": None,
                "color_fraction": None,
                "reason": str(exc),
                "approved_reward": None,
            }
        panel = Image.new("RGB", (1280, 520), "white")
        panel.paste(image, (0, 40))
        panel.paste(Image.fromarray(overlay), (640, 40))
        text = "unknown" if result["color_fraction"] is None else f"{result['color_fraction']:.1%}"
        shadow_text = (
            "unknown"
            if result.get("shadow_tolerant_color_fraction") is None
            else f"{result['shadow_tolerant_color_fraction']:.1%}"
        )
        ImageDraw.Draw(panel).text(
            (8, 10),
            f"{job['id']} row={job['row']} lift={job['lift_delta_m'] * 100:.1f}cm  conservative={text}  shadow-tolerant={shadow_text}",
            fill="black",
        )
        panel.save(output / (job["id"] + ".jpg"), quality=92)
        records.append({"job": job, "measurement": result, "visualization": f"roi/{job['id']}.jpg"})
    if audit and set(audit_by_id) != {rec["job"]["id"] for rec in records}:
        raise ValueError("audit IDs must match the complete probe set")
    report = {
        "schema": "parts_fingertip_roi_probe_v1",
        "threshold": 0.8,
        "comparison": "strictly_greater",
        "half_width_px": 6,
        "tip_method": "seed_component_y_quantile_0.01_x_median_5px",
        "color_rule": "maxRGB>=60 and RGBchroma>=45 and RGBchroma/maxRGB>=0.25",
        "shadow_tolerant_color_rule": "maxRGB>=30 and RGBchroma>=12 and RGBchroma/maxRGB>=0.20",
        "comparison_exploratory": True,
        "records": records,
        "automatic_tracking": False,
        "automatic_reward_ready": False,
        "training_executed": False,
        "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "roi-results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    cards = []
    for rec in records:
        cid = html.escape(rec["job"]["id"])
        review = audit_by_id.get(rec["job"]["id"])
        audit_text = (
            ""
            if review is None
            else f"<p><b>目视快照审核：{html.escape(review['label'])}</b> · {html.escape(review['reason'])}</p>"
        )
        cards.append(
            f'<article><h2>{cid}</h2><img loading="lazy" src="{rec["visualization"]}" alt="{cid} original and ROI">'
            + audit_text
            + f"<pre>{html.escape(json.dumps(rec['measurement'], indent=2))}</pre>"
            f'<video controls preload="none" src="{rec["job"]["clip"]}"></video></article>'
        )
    audit_summary = "" if audit is None else f"<h2>逐例核对结果</h2><p>{html.escape(audit['summary_text'])}</p>"
    (root / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>夹指尖端 ROI · 20例</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px system-ui;margin:24px;background:#eee}'
        "article{background:white;padding:16px;margin:20px 0}img{width:100%;max-width:1280px}video{width:100%;max-width:960px}"
        "pre{white-space:pre-wrap}</style><h1>夹指尖端连接区域 · 固定80%门槛 · 20例</h1>"
        "<p>左为原图，右为独立单帧夹指分割。青点：推定尖端；黄框：12px窄带；紫色：扣除夹指后的分母区域。"
        "同时列出保守色彩定义与阴影宽松定义；两者门槛均严格大于80%，第二种为首次结果后的探索性对照。"
        "彩色像素可能属于背景框；未达门槛不等于空抓。未启用跟踪，未批准奖励，未开始训练。</p>"
        + audit_summary
        + "".join(cards)
        + "</html>"
    )
    print(
        json.dumps(
            {
                "images": len(records),
                "threshold_met": sum(r["measurement"]["threshold_met"] is True for r in records),
                "invalid": sum(not r["measurement"]["geometry_valid"] for r in records),
            }
        )
    )


if __name__ == "__main__":
    main()
