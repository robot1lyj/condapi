"""Prepare fixed single-frame fingertip crops and spatial pooling weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    output = root / "crops"
    output.mkdir(exist_ok=False)
    sources = {"reference": args.reference_dataset, "evaluation": root / "sam3-eval"}
    measurements = {
        split: {r["job"]["id"]: r["measurement"] for r in json.loads((p / "roi-results.json").read_text())["records"]}
        for split, p in sources.items()
    }
    records = []
    for job in json.loads((root / "jobs.json").read_text()):
        source = root / job["image"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != job["sha256"]:
            raise ValueError("source image identity mismatch")
        m = measurements[job["split"]][job["id"]]
        if "tips_xy" not in m:
            raise ValueError("missing tips: retain unknown and prepare a new explicit protocol")
        center = np.mean(m["tips_xy"], axis=0)
        left = round(center[0]) - 128
        top = round(center[1]) - 24 - 128
        box = (left, top, left + 256, top + 256)
        image = Image.open(source).convert("RGB")
        if left < 0 or top < 0 or box[2] > image.width or box[3] > image.height:
            raise ValueError("crop leaves image; no silent padding")
        crop = image.crop(box)
        crop_path = output / (job["id"] + ".png")
        crop.save(crop_path)
        with np.load(sources[job["split"]] / "roi" / (job["id"] + ".npz")) as data:
            roi = data["roi"][top : top + 256, left : left + 256]
        weights = roi.reshape(16, 16, 16, 16).mean(axis=(1, 3)).astype(np.float32)
        if weights.sum() <= 0:
            raise ValueError("no spatial support for ROI pooling")
        weights_path = output / (job["id"] + "-weights.npz")
        np.savez_compressed(weights_path, weights=weights)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box, outline="cyan", width=2)
        draw.polygon([tuple(p) for p in m["polygon_xy"]], outline="yellow", width=2)
        overlay.save(output / (job["id"] + "-source.jpg"), quality=92)
        records.append(
            {
                **job,
                "crop": str(crop_path.relative_to(root)),
                "crop_sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest(),
                "patch_weights": str(weights_path.relative_to(root)),
                "crop_box_xyxy": box,
                "roi_measurement": m,
            }
        )
    (root / "feature-jobs.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    (root / "crop-run.json").write_text(
        json.dumps(
            {
                "images": len(records),
                "automatic_tracking": False,
                "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
