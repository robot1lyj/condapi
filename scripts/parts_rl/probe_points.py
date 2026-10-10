"""Thor-only SAM3 instance-point probe for fixed wrist-camera finger anchors."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image


def bottom_dark_anchor(rgb, side):
    """Select an inspectable dark run, without outcome or object annotations."""
    h, w = rgb.shape[:2]
    dark = (rgb.max(2) < 110) & (rgb.max(2).astype(float) - rgb.min(2) < 60)
    lo, hi = (0, int(w * 0.49)) if side == "left" else (int(w * 0.51), w)
    choices = []
    for y in (int(h * 0.94), int(h * 0.97)):
        row = np.pad(dark[y, lo:hi], (1, 1)).astype(int)
        edges = np.diff(row)
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            if b - a >= 15:
                choices.append((b - a, [int(lo + (a + b) // 2), y]))
    if not choices:
        raise ValueError("no provisional finger anchor; retain this image as uncertain")
    return max(choices, key=lambda x: x[0])[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adaptive-anchors", action="store_true")
    args = parser.parse_args()
    if platform.machine() != "aarch64":
        parser.error("model inference executes on Thor only")
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("expected Thor CUDA")
    torch.set_num_threads(4)
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if checkpoint_sha != "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e":
        raise ValueError("unreviewed checkpoint")
    prefix = "point-adaptive" if args.adaptive_anchors else "point"
    output = args.dataset / (prefix + "-predictions")
    output.mkdir(exist_ok=False)
    jobs = [j for j in json.loads((args.dataset / "jobs.json").read_text()) if j["view"] == "wrist_full"]
    started = time.monotonic()
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False,
                                      device="cuda", eval_mode=True, compile=False, enable_inst_interactivity=True)
        processor = Sam3Processor(model)
        for job in jobs:
            path = args.dataset / job["image"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != job["sha256"]:
                raise ValueError("probe image SHA mismatch")
            image = Image.open(path).convert("RGB")
            if image.size != (640, 480):
                raise ValueError("anchor probe requires recorded 640x480 wrist configuration")
            state = processor.set_image(image)
            results = []
            rgb = np.asarray(image)
            left = bottom_dark_anchor(rgb, "left") if args.adaptive_anchors else [90, 450]
            right = bottom_dark_anchor(rgb, "right") if args.adaptive_anchors else [550, 450]
            for name, points in (("left_finger", [left, right, [320, 200]]),
                                 ("right_finger", [right, left, [320, 200]])):
                masks, scores, logits = model.predict_inst(
                    state, point_coords=np.asarray(points), point_labels=np.asarray([1, 0, 0]),
                    multimask_output=True,
                )
                if not all(np.isfinite(x).all() for x in (masks, scores, logits)):
                    raise RuntimeError("nonfinite instance prediction")
                masks = np.asarray(masks, bool)
                selected = int(np.argmax(scores))
                key = f"{job['id']}-{name}"
                np.savez_compressed(output / (key + ".npz"),
                                    masks_packed=np.packbits(masks.reshape(len(masks), 640 * 480), axis=1),
                                    shape=[480, 640])
                results.append({"name": name, "points_xy": points, "point_labels": [1, 0, 0],
                                "predicted_mask_quality": np.asarray(scores).tolist(), "selected_index": selected,
                                "masks": f"{prefix}-predictions/{key}.npz"})
            record = {"job": job, "results": results, "approved_reward": None}
            (output / (job["id"] + ".json")).write_text(json.dumps(record, allow_nan=False) + "\n")
            records.append(record)
            print(json.dumps({"completed": len(records), "total": len(jobs)}), flush=True)
    report = {"schema": "parts_instance_point_probe_v1", "images": len(jobs),
              "elapsed_s": time.monotonic() - started, "checkpoint_sha256": checkpoint_sha,
              "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "anchor_method": "longest_bottom_dark_run" if args.adaptive_anchors else "fixed_camera_points",
              "anchors_provisional": True, "accuracy_measured": False, "automatic_reward_ready": False,
              "training_executed": False}
    (args.dataset / (prefix + "-summary.json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
