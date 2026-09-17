"""Read-only targeted evidence collection. Run on an allocated compute node."""

import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import sys

import av
import numpy as np
import pyarrow.parquet as pq
from openpi.training import config as configs
from openpi.training import data_loader
from PIL import Image, ImageDraw

ROOT = pathlib.Path(
    "/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train"
)
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(exist_ok=False)
manifest = json.loads((ROOT / "conversion_manifest.json").read_text())
spec = importlib.util.spec_from_file_location(
    "probe",
    "/home/wuyan/lyj/YAM/training-runs/control/yam-loader-probe-20260917/probe.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
targets = [(3227, 484, 5), (3306, 1172, 4), (1086, 1303, 12)]
result = {"targets": [], "normalizations": {}}
for eid, frame, dim in targets:
    chunk, fid = divmod(eid, 1000)
    rel = f"chunk-{chunk:03d}/file-{fid:03d}"
    path = ROOT / f"data/{rel}.parquet"
    table = pq.read_table(path)
    state = np.asarray(table["observation.state"].combine_chunks().values).reshape(
        -1, 14
    )
    action = np.asarray(table["action"].combine_chunks().values).reshape(-1, 14)
    record = dict(manifest["episodes"][eid])
    record.update(
        frame=frame,
        dimension=dim,
        parquet=str(path),
        parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    record["context"] = [
        {
            "frame": f,
            "state": float(state[f, dim]),
            "action": float(action[f, dim]),
            "delta": float(action[f, dim] - state[f, dim]),
        }
        for f in range(frame - 4, frame + 5)
    ]
    record["video_checks"] = []
    pictures = []
    for camera in ["top_rgb", "left_rgb", "right_rgb"]:
        video = ROOT / f"videos/observation.images.{camera}/{rel}.mp4"
        n = 0
        prev = None
        mono = True
        for_container = av.open(str(video))
        with for_container as container:
            for vf in container.decode(video=0):
                ts = None if vf.pts is None else float(vf.pts * vf.time_base)
                if ts is None or (prev is not None and ts <= prev):
                    mono = False
                prev = ts
                if n in [frame - 1, frame, frame + 1]:
                    pictures.append((camera, n, vf.to_image()))
                n += 1
        record["video_checks"].append(
            {
                "camera": camera,
                "decoded_frames": n,
                "expected_frames": len(state),
                "monotonic_pts": mono,
            }
        )
    canvas = Image.new("RGB", (3 * 224, 3 * 248), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (cam, f, pic) in enumerate(pictures):
        x, y = (i % 3) * 224, (i // 3) * 248
        canvas.paste(pic.resize((224, 224)), (x, y + 24))
        draw.text((x + 3, y + 5), f"{cam} ep{eid} frame{f}", fill="black")
    canvas.save(OUT / f"episode-{eid}.jpg")
    result["targets"].append(record)
for label, asset in [
    ("full", "/home/wuyan/lyj/YAM/training-assets/lego_lerobot_v1_20260907/pi05_h50"),
    ("subset", "/home/wuyan/lyj/YAM/training-assets/lego_rtc_10h_20260916/norm"),
]:
    cfg = configs.get_config("pi05_yam")
    factory = dataclasses.replace(
        cfg.data,
        repo_id=str(ROOT),
        assets=configs.AssetsConfig(assets_dir=asset, asset_id="yam"),
        base_config=dataclasses.replace(
            cfg.data.base_config, train_episodes=[t[0] for t in targets]
        ),
    )
    dc = factory.create(pathlib.Path(asset), cfg.model)
    raw = data_loader.create_torch_dataset(dc, 50, cfg.model)
    final = data_loader.transform_dataset(raw, dc)
    base = probe.unwrap_lerobot_dataset(raw)
    _ = base.hf_dataset
    ranges = probe.episode_ranges(base, [t[0] for t in targets])
    samples = []
    for eid, f, dim in targets:
        for start in sorted({max(0, f - 49), max(0, f - 1), f, f + 1}):
            idx = ranges[eid][0] + start
            r = raw[idx]
            out = final[idx]
            a = np.asarray(out["actions"])
            s = np.asarray(out["state"])
            samples.append(
                {
                    "episode": eid,
                    "frame": start,
                    "action_shape": list(a.shape),
                    "state_shape": list(s.shape),
                    "finite": bool(np.isfinite(a).all() and np.isfinite(s).all()),
                    "action_min": float(a.min()),
                    "action_max": float(a.max()),
                    "padded_action_zero": bool((a[:, 14:] == 0).all()),
                    "raw_action_target_dimension": np.asarray(r["action"])[
                        :, dim
                    ].tolist(),
                    "normalized_action_target_dimension": a[:, dim].tolist(),
                }
            )
    result["normalizations"][label] = {
        "norm_sha256": hashlib.sha256(
            (pathlib.Path(asset) / "yam/norm_stats.json").read_bytes()
        ).hexdigest(),
        "samples": samples,
    }
(OUT / "spikes.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"targets": len(targets), "windows_per_norm": 12, "output": str(OUT)}))
