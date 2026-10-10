"""Thor-only frozen DINOv3 feature extraction for independent grasp snapshots."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--skip-dense", action="store_true", help="Save only pooled ROI and CLS features")
    args = parser.parse_args()
    if platform.machine() != "aarch64":
        parser.error("neural inference executes on Thor only")
    from safetensors.torch import load_file  # noqa: PLC0415
    import timm  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("expected Thor CUDA")
    torch.set_num_threads(4)
    weight_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if weight_sha != "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040":
        raise ValueError("unreviewed DINOv3 checkpoint")
    output = args.dataset / "dinov3"
    output.mkdir(exist_ok=False)
    jobs_path = args.dataset / "feature-jobs.json"
    jobs = json.loads(jobs_path.read_text())
    started = time.monotonic()
    model = timm.create_model("vit_small_patch16_dinov3", pretrained=False, num_classes=0)
    model.load_state_dict(load_file(str(args.weights)), strict=True)
    model.eval().cuda()
    model.requires_grad_(False)  # noqa: FBT003
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    cls_features, roi_features, dense_features, latencies = [], [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(jobs), 8):
            group = jobs[start : start + 8]
            images, weights = [], []
            for job in group:
                path = args.dataset / job["crop"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != job["crop_sha256"]:
                    raise ValueError("crop identity mismatch")
                rgb = np.asarray(Image.open(path).convert("RGB"))
                if rgb.shape != (256, 256, 3):
                    raise ValueError("expected exact 256x256 crop")
                with np.load(args.dataset / job["patch_weights"]) as saved:
                    weight = saved["weights"]
                if weight.shape != (16, 16) or not np.isfinite(weight).all() or weight.sum() <= 0:
                    raise ValueError("invalid ROI patch weights")
                images.append(rgb)
                weights.append(weight.reshape(-1))
            inputs = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255
            inputs = (inputs - mean) / std
            torch.cuda.synchronize()
            tick = time.monotonic()
            tokens = model.forward_features(inputs)
            torch.cuda.synchronize()
            latencies.append(time.monotonic() - tick)
            patches = tokens[:, model.num_prefix_tokens :].float()
            if patches.shape[1:] != (256, 384) or not torch.isfinite(tokens).all():
                raise RuntimeError("unexpected/nonfinite DINOv3 features")
            w = torch.from_numpy(np.stack(weights)).float().cuda()
            pooled = (patches * w[:, :, None]).sum(1) / w.sum(1)[:, None]
            cls_features.append(tokens[:, 0].float().cpu().numpy())
            roi_features.append(pooled.cpu().numpy())
            if not args.skip_dense:
                dense_features.append(patches.cpu().numpy())
            print(json.dumps({"completed": start + len(group), "total": len(jobs)}), flush=True)
    np.savez_compressed(
        output / "features.npz",
        ids=np.array([j["id"] for j in jobs]),
        cls=np.concatenate(cls_features),
        roi=np.concatenate(roi_features),
        **({"patches": np.concatenate(dense_features)} if dense_features else {}),
    )
    report = {
        "schema": "parts_frozen_dinov3_probe_v1",
        "images": len(jobs),
        "dense_features_saved": not args.skip_dense,
        "model": "timm/vit_small_patch16_dinov3.lvd1689m",
        "revision": "3bf4720a82ec2066db88137180ff1f83a675cef0",
        "weights_sha256": weight_sha,
        "model_params_dtype": str(next(model.parameters()).dtype),
        "autocast": "bfloat16",
        "versions": {k: importlib.metadata.version(k) for k in ["torch", "timm", "safetensors"]},
        "elapsed_s": time.monotonic() - started,
        "batch_inference_s": latencies,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "jobs_sha256": hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
        "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "automatic_tracking": False,
        "gradient_updates": 0,
        "training_executed": False,
        "automatic_reward_ready": False,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
