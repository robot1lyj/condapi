"""Thor-only SAM3 visual exemplars at provisional wrist finger regions."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image


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

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("expected Thor CUDA")
    torch.set_num_threads(4)
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if checkpoint_sha != "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e":
        raise ValueError("unreviewed checkpoint")
    output = args.dataset / "visual-predictions"
    output.mkdir(exist_ok=False)
    jobs = [j for j in json.loads((args.dataset / "jobs.json").read_text()) if j["view"] == "wrist_full"]
    # Provisional fixed camera ROIs, not per-frame ground-truth boxes.
    boxes = {"left_finger": [0.245, 0.82, 0.49, 0.36],
             "right_finger": [0.755, 0.82, 0.49, 0.36]}
    started = time.monotonic()
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False,
                                      device="cuda", eval_mode=True, compile=False)
        processor = Sam3Processor(model, confidence_threshold=0.10)
        for job in jobs:
            path = args.dataset / job["image"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != job["sha256"]:
                raise ValueError("probe image SHA mismatch")
            image = Image.open(path).convert("RGB")
            state = processor.set_image(image)
            results = []
            for name, box in boxes.items():
                processor.reset_all_prompts(state)
                state = processor.add_geometric_prompt(box=box, label=True, state=state)
                if any(not bool(torch.isfinite(state[k]).all()) for k in ("boxes", "scores", "masks_logits")):
                    raise RuntimeError("nonfinite selected prediction")
                masks = state["masks"].squeeze(1).cpu().numpy()
                probabilities = state["scores"].float().cpu().numpy()
                corners = state["boxes"].float().cpu().numpy()
                objects = [{"score": float(s), "box": b.tolist(), "mask_index": i,
                            "area": int(m.sum())}
                           for i, (s, b, m) in enumerate(zip(probabilities, corners, masks, strict=True))]
                key = f"{job['id']}-{name}"
                packed = np.packbits(masks.reshape(len(masks), image.height * image.width), axis=1)
                np.savez_compressed(output / (key + ".npz"), masks_packed=packed,
                                    shape=[image.height, image.width])
                results.append({"name": name, "prompt": "visual", "normalized_box_cxcywh": box,
                                "max_selected_score": max((o["score"] for o in objects), default=0),
                                "objects": objects, "masks": f"visual-predictions/{key}.npz"})
            record = {"job": job, "results": results, "approved_reward": None}
            (output / (job["id"] + ".json")).write_text(json.dumps(record, allow_nan=False) + "\n")
            records.append(record)
            print(json.dumps({"completed": len(records), "total": len(jobs)}), flush=True)
    report = {"schema": "parts_visual_box_probe_v1", "images": len(jobs), "normalized_boxes": boxes,
              "images_with_any_score_ge_0.50": sum(any(r["max_selected_score"] >= 0.5 for r in x["results"])
                                                    for x in records),
              "elapsed_s": time.monotonic() - started, "checkpoint_sha256": checkpoint_sha,
              "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "accuracy_measured": False, "automatic_reward_ready": False, "training_executed": False}
    (args.dataset / "visual-summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
