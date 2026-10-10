"""Thor-only descriptive SAM3 prompt comparison; never assigns rewards."""

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image

PROMPTS = [
    "robot gripper fingers",
    "robot gripper",
    "black gripper",
    "black triangular gripper fingers",
    "black triangular shapes",
    "black triangular plastic wedges",
    "black ribbed triangular fingers",
    "two black tapered fingers",
    "black wedge-shaped jaws",
    "black ridged plastic jaws",
    "black triangle",
    "LEGO brick",
    "plastic building block",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if platform.machine() != "aarch64":
        parser.error("model inference executes on Thor only")
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("expected Thor CUDA")
    torch.set_num_threads(4)
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if checkpoint_sha != "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e":
        raise ValueError("unreviewed checkpoint")
    output = args.dataset / "predictions"
    output.mkdir(exist_ok=False)
    jobs = json.loads((args.dataset / "jobs.json").read_text())
    pipeline_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.monotonic()
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False,
                                      device="cuda", eval_mode=True, compile=False)
        processor = Sam3Processor(model, confidence_threshold=0.10)
        texts = [model.backbone.forward_text([p], device="cuda") for p in PROMPTS]
        for begin in range(0, len(jobs), 4):
            batch = jobs[begin:begin + 4]
            images = []
            for job in batch:
                path = args.dataset / job["image"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != job["sha256"]:
                    raise ValueError("probe image SHA mismatch")
                images.append(Image.open(path).convert("RGB"))
            state = processor.set_image_batch(images)
            size = len(images)
            find = replace(processor.find_stage, img_ids=torch.arange(size, device="cuda"),
                           text_ids=torch.zeros(size, dtype=torch.long, device="cuda"))
            groups = [[] for _ in images]
            for pi, text in enumerate(texts):
                state["backbone_out"].update(text)
                raw = model.forward_grounding(backbone_out=state["backbone_out"], find_input=find,
                                             geometric_prompt=model._get_dummy_prompt(size), find_target=None)  # noqa: SLF001
                if any(not bool(torch.isfinite(raw[k]).all()) for k in
                       ("pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec")):
                    raise RuntimeError("nonfinite prediction")
                scores = (raw["pred_logits"].sigmoid() * raw["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
                for bi, image in enumerate(images):
                    keep = scores[bi] > 0.10
                    masks = functional.interpolate(raw["pred_masks"][bi, keep].unsqueeze(1),
                                                   (image.height, image.width), mode="bilinear", align_corners=False)
                    masks = (masks.sigmoid() > 0.5).squeeze(1).cpu().numpy()
                    boxes = raw["pred_boxes"][bi, keep].float().cpu().numpy()
                    probabilities = scores[bi, keep].float().cpu().numpy()
                    objects = []
                    for mi, (mask, box, score) in enumerate(zip(masks, boxes, probabilities, strict=True)):
                        cx, cy, bw, bh = box
                        xyxy = [(cx - bw / 2) * image.width, (cy - bh / 2) * image.height,
                                (cx + bw / 2) * image.width, (cy + bh / 2) * image.height]
                        objects.append({"score": float(score), "box": [float(x) for x in xyxy],
                                        "mask_index": mi, "area": int(mask.sum())})
                    key = f"{batch[bi]['id']}-p{pi:02d}"
                    packed = np.packbits(masks.reshape(len(masks), image.height * image.width), axis=1)
                    np.savez_compressed(output / (key + ".npz"), masks_packed=packed,
                                        shape=[image.height, image.width])
                    groups[bi].append({"prompt_index": pi, "prompt": PROMPTS[pi],
                                       "max_score": float(scores[bi].max()), "objects": objects,
                                       "masks": f"predictions/{key}.npz"})
            for job, results in zip(batch, groups, strict=True):
                record = {"job": job, "prompts": results, "approved_reward": None}
                (output / (job["id"] + ".json")).write_text(json.dumps(record, allow_nan=False) + "\n")
                records.append(record)
            print(json.dumps({"completed": min(begin + 4, len(jobs)), "total": len(jobs),
                              "elapsed_s": time.monotonic() - started}), flush=True)
    counts = defaultdict(lambda: defaultdict(int))
    for record in records:
        for result in record["prompts"]:
            counts[result["prompt"]][record["job"]["view"] + "_max_ge_0.50"] += result["max_score"] >= 0.50
    report = {"schema": "parts_descriptive_prompt_probe_v1", "jobs": len(jobs), "prompts": PROMPTS,
              "counts": dict(counts), "elapsed_s": time.monotonic() - started,
              "checkpoint_sha256": checkpoint_sha, "pipeline_sha256": pipeline_sha,
              "accuracy_measured": False, "automatic_reward_ready": False, "training_executed": False}
    (args.dataset / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
