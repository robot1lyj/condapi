"""Server-GPU-only, frozen official ResNet18 features from causal raw frames."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import av
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Frozen model forward is allowed only on the allocated server GPU")
    output = args.replay / "features.npy"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.fc = nn.Identity()
    model.requires_grad_(requires_grad=False).eval().cuda()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    jobs = json.loads((args.replay / "frame-jobs.json").read_text())
    obs = json.loads((args.replay / "observations.json").read_text())
    by_video, indices = defaultdict(dict), {}
    for i, job in enumerate(jobs):
        by_video[job["video"]][job["frame"]] = i
        indices[job["key"]] = i
    features = np.empty((len(jobs), 512), dtype=np.float32)
    seen, batch, positions = set(), [], []
    started = time.monotonic()

    @torch.inference_mode()
    def flush():
        if batch:
            encoded = model(torch.stack(batch).cuda()).cpu().numpy()
            if not np.isfinite(encoded).all():
                raise ValueError("nonfinite frozen features")
            features[positions] = encoded
            seen.update(positions)
            batch.clear()
            positions.clear()

    manifest = json.loads((args.replay / "manifest.json").read_text())
    for number, (video, picks) in enumerate(sorted(by_video.items()), 1):
        path = args.raw_root / video
        if sha256(path) != manifest["raw_sha256"]["raw/" + video]:
            raise ValueError("server raw video hash differs: " + video)
        with av.open(str(path)) as container:
            container.streams.video[0].codec_context.thread_count = 2
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in picks:
                    batch.append(transform(frame.to_image()))
                    positions.append(picks[frame_index])
                    if len(batch) >= args.batch_size:
                        flush()
        flush()
        print(
            json.dumps(
                {"video": number, "total": len(by_video), "images": len(seen), "seconds": time.monotonic() - started}
            ),
            flush=True,
        )
    if len(seen) != len(jobs):
        raise ValueError("some requested image frames were not decoded")
    grouped = features[np.array([[indices[k] for k in o["frames"]] for o in obs])]
    grouped /= np.maximum(np.linalg.norm(grouped, axis=-1, keepdims=True), 1e-12)
    np.save(output, grouped.reshape(len(obs), -1))
    write_json(
        args.replay / "feature-manifest.json",
        {
            "encoder": "torchvision resnet18 IMAGENET1K_V1, frozen FP32, fc removed",
            "weights_sha256": sha256(args.weights),
            "torch": torch.__version__,
            "preprocessing": "official weight transforms, Resize256 CenterCrop224 ImageNet normalization; per-view L2",
            "views": ["top", "left", "right"],
            "observations": len(obs),
            "images": len(jobs),
            "features_sha256": sha256(output),
            "frame_jobs_sha256": sha256(args.replay / "frame-jobs.json"),
            "replay_sha256": sha256(args.replay / "replay.npz"),
            "seconds": time.monotonic() - started,
            "gpu": torch.cuda.get_device_name(),
            "reward_model_features_reused": False,
            "future_frames_in_observation": False,
        },
    )


if __name__ == "__main__":
    main()
