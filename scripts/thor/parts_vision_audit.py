"""Thor-only dependency/CUDA audit; optional inference, never a training loop.

Random-weight inference exercises kernels only and cannot validate detections
or issue RL rewards. Real-weight semantic validation is a separate gate.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-kernel-check", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="LEGO brick")
    parser.add_argument("--predictions-dir", type=Path)
    args = parser.parse_args()
    if args.random_kernel_check and args.checkpoint:
        parser.error("random and pretrained inference are distinct audit modes")
    if args.output.exists():
        parser.error("write a new audit output; do not overwrite previous evidence")
    if args.predictions_dir and (not args.checkpoint or not args.image):
        parser.error("prediction evidence requires pretrained weights and a source image")
    if args.predictions_dir and args.predictions_dir.exists():
        parser.error("write predictions to a new directory")
    if platform.machine() != "aarch64":
        parser.error("this audit executes only on Thor ARM64")

    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from PIL import ImageDraw  # noqa: PLC0415
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
    from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torchvision  # noqa: PLC0415

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("expected a CUDA-enabled NVIDIA Thor")
    torch.set_num_threads(4)
    report = {
        "schema": "parts_vision_environment_audit_v1", "architecture": platform.machine(),
        "python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "packages": {name: importlib.metadata.version(name) for name in (
            "sam3", "numpy", "timm", "ftfy", "iopath", "av", "h5py", "setuptools", "triton",
        )},
        "pretrained_weights_loaded": False, "reward_generation_ready": False,
        "robot_connected": False, "training_executed": False,
    }
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False)
    report["pip_check"] = {"exit_code": result.returncode, "output": result.stdout + result.stderr}
    if result.returncode:
        raise RuntimeError("pip dependency check failed: " + report["pip_check"]["output"])
    with torch.inference_mode():
        x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        product = x @ x.T
        if not torch.isfinite(product).all():
            raise RuntimeError("nonfinite CUDA matrix multiplication")
        nms = torchvision.ops.nms(
            torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], device="cuda", dtype=torch.float32),
            torch.tensor([0.9, 0.8], device="cuda"), 0.5,
        )
        report["cuda_checks"] = {"matmul_finite": True, "torchvision_nms": nms.cpu().tolist()}
        if args.random_kernel_check or args.checkpoint:
            if args.checkpoint and not args.checkpoint.is_file():
                raise FileNotFoundError(args.checkpoint)
            torch.manual_seed(0)
            start = time.perf_counter()
            model = build_sam3_image_model(
                checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
                load_from_HF=False, device="cuda", eval_mode=True, compile=False,
            )
            # Upstream's image notebook uses BF16 autocast; fused MLP kernels
            # produce BF16. Preserve parameter storage, do not convert weights.
            processor = Sam3Processor(model, confidence_threshold=0.5)
            image = Image.open(args.image).convert("RGB") if args.image else Image.fromarray(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )
            torch.cuda.synchronize()
            loaded = time.perf_counter()
            raw_checks = {}
            forward_grounding = model.forward_grounding

            def checked_grounding(*positional, **keywords):
                raw = forward_grounding(*positional, **keywords)
                for key in ("pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec"):
                    tensor = raw[key]
                    raw_checks[key] = {
                        "shape": list(tensor.shape), "numel": tensor.numel(),
                        "finite": bool(torch.isfinite(tensor).all()),
                    }
                return raw

            # Audit unfiltered tensors: an empty detection set must not make
            # the finite-output check pass vacuously. Restore after this call.
            model.forward_grounding = checked_grounding
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = processor.set_image(image)
                    output = processor.set_text_prompt(state=state, prompt=args.prompt)
            finally:
                model.forward_grounding = forward_grounding
            torch.cuda.synchronize()
            report["inference"] = {
                "mode": "pretrained" if args.checkpoint else "random_weights_kernel_check_only",
                "model_load_s": loaded - start, "first_inference_s": time.perf_counter() - loaded,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                "compute_mode": "upstream_bfloat16_autocast_original_parameter_storage",
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "prompt": args.prompt, "raw_outputs": raw_checks,
                "outputs": {key: {"shape": list(output[key].shape), "finite": bool(torch.isfinite(output[key]).all())}
                            for key in ("masks", "boxes", "scores")},
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "semantic_accuracy_validated": False,
            }
            if not all(v["finite"] for v in report["inference"]["outputs"].values()):
                raise RuntimeError("nonfinite SAM3 inference")
            if not raw_checks or not all(v["finite"] and v["numel"] for v in raw_checks.values()):
                raise RuntimeError("empty or nonfinite raw SAM3 predictions")
            if args.checkpoint:
                with args.checkpoint.open("rb") as file:
                    report["checkpoint_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
                report["pretrained_weights_loaded"] = True
            if args.predictions_dir:
                args.predictions_dir.mkdir(parents=True, exist_ok=False)
                masks = output["masks"].cpu().numpy()
                boxes = output["boxes"].float().cpu().numpy()
                scores = output["scores"].float().cpu().numpy()
                np.savez_compressed(args.predictions_dir / "predictions.npz", masks=masks, boxes=boxes, scores=scores)
                overlay = image.convert("RGBA")
                colors = ((255, 90, 50), (40, 220, 100), (50, 130, 255), (240, 180, 40))
                for index, mask in enumerate(masks):
                    color = colors[index % len(colors)]
                    alpha = Image.fromarray(np.asarray(mask.squeeze(), dtype=np.uint8) * 75)
                    tint = Image.new("RGBA", image.size, (*color, 0))
                    tint.putalpha(alpha)
                    overlay = Image.alpha_composite(overlay, tint)
                draw = ImageDraw.Draw(overlay)
                for index, (box, score) in enumerate(zip(boxes, scores, strict=True)):
                    color = colors[index % len(colors)]
                    draw.rectangle(box.tolist(), outline=(*color, 255), width=2)
                    draw.text((float(box[0]), max(0.0, float(box[1]) - 12)), f"{index}: {score:.2f}", fill=(*color, 255))
                overlay.convert("RGB").save(args.predictions_dir / "overlay.png")
                report["prediction_evidence"] = {
                    "source_image": str(args.image),
                    "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
                    "detections": len(scores), "boxes_xyxy": boxes.tolist(), "scores": scores.tolist(),
                    "directory": str(args.predictions_dir), "grasp_reward": None,
                    "reason": "object masks alone do not establish grasp success",
                }
    report["status"] = "environment_checked_not_reward_validated"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
