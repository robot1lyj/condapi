"""Thor-only native SAM3 batch inference for an offline review dataset, no training."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image


def geometry(image, groups):
    """Inspectable geometric evidence, not an approved held/empty classifier."""
    h, w = image.shape[:2]
    gray = image.astype(float).mean(2)
    chroma = image.max(2).astype(float) - image.min(2)
    dark = (image.max(2) < 90) & (chroma < 40)
    gap, left, right = (np.zeros((h, w), bool) for _ in range(3))
    for y in range(int(h * 0.65), int(h * 0.97)):
        left_pixels = np.flatnonzero(dark[y, :w // 2 - 5])
        right_pixels = np.flatnonzero(dark[y, w // 2 + 5:]) + w // 2 + 5
        if not len(left_pixels) or not len(right_pixels):
            continue
        a, b = int(left_pixels[-1]), int(right_pixels[0])
        if 8 <= b - a <= w * 0.55:
            gap[y, a + 3:b - 2] = True
            left[y, max(0, a - 12):a + 2] = True
            right[y, b - 1:min(w, b + 13)] = True
    bins = np.zeros((h, w), bool)
    for obj in groups["bin"]:
        if obj["score"] >= 0.50:
            bins |= obj["mask"]
    gap_area = int(gap.sum())
    clear_geometry = gap_area >= 800 and np.count_nonzero(gap.any(1)) >= h * 0.12
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    sharpness = float(np.mean(lap**2))
    floor_fraction = float(((gray > 100) & (chroma < 45) & gap).sum() / max(1, gap_area))
    bin_fraction = float((bins & gap).sum() / max(1, gap_area))
    held, ambiguous, summaries = [], [], []
    for obj in groups["brick"]:
        mask = obj["mask"]
        area = int(mask.sum())
        yy, xx = np.nonzero(mask)
        box = obj["box"]
        x0, y0, x1, y1 = (int(v) for v in box)
        rectangle = np.zeros((h, w), bool)
        rectangle[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = True
        contacts = bool((rectangle & left).sum() >= 12 and (rectangle & right).sum() >= 12)
        overlap = float((mask & gap).sum() / max(1, area))
        occupied = float((mask & gap).sum() / max(1, gap_area))
        bin_overlap = float((mask & bins).sum() / max(1, area))
        item = {"score": obj["score"], "box": box, "area": area,
                "centroid": [float(xx.mean()), float(yy.mean())] if area else None,
                "gap_overlap": overlap, "gap_occupied": occupied, "bin_overlap": bin_overlap,
                "contacts_both_finger_regions": contacts}
        summaries.append(item)
        if clear_geometry and occupied >= 0.05 and bin_overlap < 0.25:
            ambiguous.append(item)
        if (clear_geometry and contacts and obj["score"] >= 0.60 and area >= 300
                and overlap >= 0.12 and occupied >= 0.12 and bin_overlap < 0.15
                and box[2] - box[0] < w * 0.60 and sharpness >= 25):
            held.append(item)
    gripper_seen = any(o["score"] >= 0.60 for o in groups["gripper"])
    empty = (clear_geometry and gripper_seen and not ambiguous and floor_fraction >= 0.90
             and bin_fraction < 0.10 and sharpness >= 40)
    return {"holding_candidate": bool(held), "empty_candidate": bool(empty),
            "held_objects": held, "brick_objects": summaries, "gripper_seen": gripper_seen,
            "clear_finger_gap_geometry": bool(clear_geometry), "gap_area": gap_area,
            "floor_like_gap_fraction": floor_fraction, "bin_gap_fraction": bin_fraction,
            "sharpness": sharpness, "note": "Photometric/geometry predicates are provisional review evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-name", default="vision")
    parser.add_argument("--verify-batch", action="store_true")
    args = parser.parse_args()
    if platform.machine() != "aarch64":
        parser.error("neural inference executes on Thor only")
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("expected NVIDIA Thor CUDA")
    torch.set_num_threads(4)
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if checkpoint_sha != "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e":
        raise ValueError("unreviewed checkpoint")
    output_dir = args.dataset / args.output_name
    output_dir.mkdir(exist_ok=True)
    pipeline_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jobs = json.loads((args.dataset / "image-jobs.json").read_text())
    if args.limit:
        jobs = jobs[:args.limit]
    pending = []
    for job in jobs:
        path = output_dir / (job["frame_key"] + ".json")
        if path.exists():
            old = json.loads(path.read_text())
            if (old["image_sha256"] != job["sha256"] or old["checkpoint_sha256"] != checkpoint_sha
                    or old.get("pipeline_sha256") != pipeline_sha):
                raise RuntimeError("existing prediction identity mismatch")
        else:
            pending.append(job)
    prompts = {"brick": "LEGO brick", "bin": "plastic storage bin", "gripper": "robot gripper fingers"}
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False,
                                      device="cuda", eval_mode=True, compile=False)
        processor = Sam3Processor(model, confidence_threshold=0.35)
        text = {name: model.backbone.forward_text([prompt], device="cuda") for name, prompt in prompts.items()}

        def infer(images):
            state = processor.set_image_batch(images)
            size = len(images)
            find = replace(processor.find_stage, img_ids=torch.arange(size, device="cuda"),
                           text_ids=torch.zeros(size, dtype=torch.long, device="cuda"))
            groups = [{name: [] for name in prompts} for _ in images]
            for name, language in text.items():
                state["backbone_out"].update(language)
                raw = model.forward_grounding(backbone_out=state["backbone_out"], find_input=find,
                                             geometric_prompt=model._get_dummy_prompt(size), find_target=None)  # noqa: SLF001
                if any(not bool(torch.isfinite(raw[key]).all()) for key in
                       ("pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec")):
                    raise RuntimeError("nonfinite unfiltered SAM3 prediction")
                scores = (raw["pred_logits"].sigmoid() * raw["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
                for index, image in enumerate(images):
                    keep = scores[index] > (0.35 if name == "brick" else 0.50)
                    masks = functional.interpolate(raw["pred_masks"][index, keep].unsqueeze(1),
                                                   (image.height, image.width), mode="bilinear", align_corners=False)
                    masks = (masks.sigmoid() > 0.5).squeeze(1).cpu().numpy()
                    boxes = raw["pred_boxes"][index, keep].float().cpu().numpy()
                    probabilities = scores[index, keep].float().cpu().numpy()
                    for mask, box, score in zip(masks, boxes, probabilities, strict=True):
                        cx, cy, bw, bh = box
                        corners = [(cx - bw / 2) * image.width, (cy - bh / 2) * image.height,
                                   (cx + bw / 2) * image.width, (cy + bh / 2) * image.height]
                        groups[index][name].append({"score": float(score), "box": [float(x) for x in corners], "mask": mask})
            return groups

        parity = None
        if args.verify_batch and pending:
            images = [Image.open(args.dataset / job["image"]).convert("RGB") for job in pending[:3]]
            multiple = infer(images)
            deltas = []
            for index, image in enumerate(images):
                single = infer([image])[0]
                for name in prompts:
                    # Compare high-confidence detections by their box, not query order.
                    for obj in single[name]:
                        if obj["score"] < 0.60:
                            continue
                        distances = [max(abs(a - b) for a, b in zip(obj["box"], o["box"], strict=True)) for o in multiple[index][name]]
                        if not distances or min(distances) > 12:
                            raise RuntimeError("batch/single high-confidence box mismatch")
                        deltas.append(float(min(distances)))
            parity = {"images": len(images), "matched_detections": len(deltas), "max_box_delta_px": max(deltas, default=0)}
            print(json.dumps({"batch_parity": parity}), flush=True)
        completed = 0
        for begin in range(0, len(pending), args.batch_size):
            batch = pending[begin:begin + args.batch_size]
            images = [Image.open(args.dataset / job["image"]).convert("RGB") for job in batch]
            groups = infer(images)
            for job, image, group in zip(batch, images, groups, strict=True):
                if hashlib.sha256((args.dataset / job["image"]).read_bytes()).hexdigest() != job["sha256"]:
                    raise RuntimeError("source image SHA mismatch")
                masks, descriptors = [], {}
                for name, objects in group.items():
                    descriptors[name] = []
                    for obj in objects:
                        descriptors[name].append({"mask_index": len(masks), "score": obj["score"], "box": obj["box"]})
                        masks.append(obj["mask"])
                packed = np.packbits(np.stack(masks).reshape(len(masks), -1), axis=1) if masks else np.empty((0, 38400), np.uint8)
                np.savez_compressed(output_dir / (job["frame_key"] + ".npz"), masks_packed=packed, shape=[image.height, image.width])
                record = {"frame_key": job["frame_key"], "image_sha256": job["sha256"],
                          "checkpoint_sha256": checkpoint_sha, "pipeline_sha256": pipeline_sha,
                          "prompts": prompts, "detections": descriptors,
                          "geometry": geometry(np.asarray(image), group), "approved_reward": None}
                (output_dir / (job["frame_key"] + ".json")).write_text(json.dumps(record, allow_nan=False) + "\n")
            completed += len(batch)
            print(json.dumps({"completed": completed, "pending_total": len(pending),
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    report = {"schema": "parts_review_vision_run_v1", "jobs": len(jobs), "new_predictions": len(pending),
              "checkpoint_sha256": checkpoint_sha, "batch_size": args.batch_size, "parity": parity,
              "pipeline_sha256": pipeline_sha,
              "elapsed_s": time.monotonic() - started, "training_executed": False,
              "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}), "prompts": prompts}
    report_path = args.dataset / (args.output_name + "-run.json")
    if report_path.exists():
        report_path = args.dataset / (args.output_name + f"-run-{time.time_ns()}.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
