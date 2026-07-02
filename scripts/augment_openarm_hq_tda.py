"""Build an OpenArm HQ TDA-augmented LeRobot v2.1 dataset.

The script is intentionally v2.1-oriented: it expects one parquet/video per
episode and writes a new version directory without modifying the source.
Video transforms prefer ffmpeg NVENC/CUDA when available; parquet and metadata
rewrites remain CPU-bound because pyarrow/pandas own that path.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import concurrent.futures
import datetime as dt
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
VECTOR_KEYS = ("action", "observation.state")
SCALAR_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
RIGHT_ARM = slice(0, 8)
LEFT_ARM = slice(8, 16)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_data_path(info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(info["data_path"].format(episode_chunk=chunk, episode_index=episode_index))


def _format_video_path(info: dict[str, Any], episode_index: int, video_key: str) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(
        info["video_path"].format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)
    )


def _parse_split(info: dict[str, Any], split: str) -> list[int]:
    if split == "all":
        return list(range(int(info["total_episodes"])))
    split_spec = info.get("splits", {}).get(split)
    if not split_spec:
        raise ValueError(f"Split {split!r} not found in meta/info.json")
    if ":" not in split_spec:
        raise ValueError(f"Only contiguous LeRobot splits are supported, got {split_spec!r}")
    start_text, end_text = split_spec.split(":", 1)
    return list(range(int(start_text), int(end_text)))


def _copy_or_link(src: pathlib.Path, dst: pathlib.Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    raise ValueError(f"Unsupported copy mode: {mode}")


def _swap_openarm_16d(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (16,):
        raise ValueError(f"Expected OpenArm 16D state/action, got shape {arr.shape}")
    return np.concatenate([arr[LEFT_ARM], arr[RIGHT_ARM]]).astype(np.float32)


def _prepare_episode_dataframe(
    src_path: pathlib.Path,
    *,
    new_episode_index: int,
    start_global_index: int,
    fps: int,
    extraction_factor: int = 1,
    mirror: bool = False,
) -> pd.DataFrame:
    episode_frame = pd.read_parquet(src_path)
    episode_frame = episode_frame.iloc[::extraction_factor].copy() if extraction_factor > 1 else episode_frame.copy()
    episode_frame = episode_frame.reset_index(drop=True)

    if mirror:
        for key in VECTOR_KEYS:
            episode_frame[key] = episode_frame[key].map(_swap_openarm_16d)

    n_rows = len(episode_frame)
    frame_index = np.arange(n_rows, dtype=np.int64)
    episode_frame["episode_index"] = np.full(n_rows, new_episode_index, dtype=np.int64)
    episode_frame["frame_index"] = frame_index
    episode_frame["index"] = np.arange(start_global_index, start_global_index + n_rows, dtype=np.int64)
    episode_frame["timestamp"] = (frame_index / float(fps)).astype(np.float32)
    if "task_index" in episode_frame.columns:
        episode_frame["task_index"] = episode_frame["task_index"].astype(np.int64)
    return episode_frame


def _numeric_stats_from_dataframe(df: pd.DataFrame) -> dict[str, dict[str, list[float] | list[int]]]:
    stats: dict[str, dict[str, list[float] | list[int]]] = {}
    for key in (*VECTOR_KEYS, *SCALAR_KEYS):
        if key not in df.columns:
            continue
        if key in VECTOR_KEYS:
            values = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in df[key].to_numpy()])
        else:
            values = df[key].to_numpy(dtype=np.float64).reshape(-1, 1)
        stats[key] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [int(values.shape[0])],
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q10": np.quantile(values, 0.10, axis=0).tolist(),
            "q50": np.quantile(values, 0.50, axis=0).tolist(),
            "q90": np.quantile(values, 0.90, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return stats


def _episode_stats_for_transformed(
    df: pd.DataFrame,
    source_stats: dict[str, Any],
    *,
    mirror: bool,
) -> dict[str, Any]:
    stats = _numeric_stats_from_dataframe(df)
    for key in VIDEO_KEYS:
        if mirror and key == "observation.images.left_wrist":
            source_key = "observation.images.right_wrist"
        elif mirror and key == "observation.images.right_wrist":
            source_key = "observation.images.left_wrist"
        else:
            source_key = key
        if source_key in source_stats:
            stats[key] = source_stats[source_key]
    return stats


def _ffmpeg_has(ffmpeg: pathlib.Path, args: list[str], needles: tuple[str, ...]) -> dict[str, bool]:
    result = subprocess.run([str(ffmpeg), *args], check=False, text=True, capture_output=True)
    haystack = result.stdout + result.stderr
    return {needle: needle in haystack for needle in needles}


def _encoder_smoke_test(
    ffmpeg: pathlib.Path,
    encoder: str,
    *,
    gpu_id: int,
    preset: str,
    quality: int,
) -> bool:
    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=size=64x64:rate=1:duration=1",
        "-an",
        "-c:v",
        encoder,
    ]
    if encoder.endswith("_nvenc"):
        cmd += ["-gpu", str(gpu_id), "-preset", preset, "-cq", str(quality)]
    else:
        cmd += ["-preset", "ultrafast", "-crf", str(quality)]
    cmd += ["-f", "null", "-"]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return result.returncode == 0


def _select_encoder(
    ffmpeg: pathlib.Path,
    requested: str,
    *,
    gpu_id: int,
    preset: str,
    quality: int,
    require_gpu: bool,
) -> str:
    if requested != "auto":
        if not _encoder_smoke_test(ffmpeg, requested, gpu_id=gpu_id, preset=preset, quality=quality):
            raise RuntimeError(f"Requested video encoder {requested!r} is listed but failed an ffmpeg smoke test")
        if require_gpu and not requested.endswith("_nvenc"):
            raise RuntimeError(f"GPU video was required, but requested encoder is {requested!r}")
        return requested
    encoders = _ffmpeg_has(ffmpeg, ["-hide_banner", "-encoders"], ("av1_nvenc", "h264_nvenc", "hevc_nvenc"))
    for candidate in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
        if encoders[candidate] and _encoder_smoke_test(
            ffmpeg, candidate, gpu_id=gpu_id, preset=preset, quality=quality
        ):
            return candidate
    if require_gpu:
        raise RuntimeError("No working NVENC encoder found. Re-run with --no-require-gpu-video to allow CPU encode.")
    return "libx264"


def _build_filter(extraction_factor: int, fps: int, *, hflip: bool) -> str:
    filters: list[str] = []
    if extraction_factor > 1:
        filters.append(f"select='not(mod(n\\,{extraction_factor}))'")
    if hflip:
        filters.append("hflip")
    if extraction_factor > 1:
        filters.append(f"setpts=N/({fps}*TB)")
    return ",".join(filters)


def _run_video_job(job: dict[str, Any]) -> None:
    src = pathlib.Path(job["src"])
    dst = pathlib.Path(job["dst"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    encoder = job["encoder"]

    def build_cmd(*, use_hwaccel: bool) -> list[str]:
        cmd = [
            str(job["ffmpeg"]),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]
        if use_hwaccel:
            cmd += ["-hwaccel", "cuda", "-hwaccel_device", str(job["gpu_id"])]
        cmd += ["-i", str(src)]

        video_filter = _build_filter(int(job["extraction_factor"]), int(job["fps"]), hflip=bool(job["hflip"]))
        if video_filter:
            cmd += ["-vf", video_filter]

        cmd += ["-an", "-c:v", encoder]
        if encoder.endswith("_nvenc"):
            cmd += ["-gpu", str(job["gpu_id"]), "-preset", str(job["preset"]), "-cq", str(job["quality"])]
        else:
            cmd += ["-preset", "veryfast", "-crf", str(job["quality"])]
        cmd += ["-pix_fmt", "yuv420p", str(dst)]
        return cmd

    use_hwaccel = bool(job["use_gpu_decode"])
    try:
        subprocess.run(build_cmd(use_hwaccel=use_hwaccel), check=True)
    except subprocess.CalledProcessError:
        if not use_hwaccel:
            raise
        if dst.exists():
            dst.unlink()
        # Some ffmpeg builds reject CUDA frames for CPU-only filters such as
        # select/hflip. Keep NVENC for the expensive encode path in the retry.
        subprocess.run(build_cmd(use_hwaccel=False), check=True)


def _write_manifest(
    path: pathlib.Path,
    *,
    dataset_id: str,
    source: pathlib.Path,
    source_split: str,
    source_episodes: int,
    time_episodes: int,
    mirror_episodes: int,
    total_episodes: int,
    total_frames: int,
    ffmpeg: pathlib.Path,
    encoder: str,
    video_workers: int,
) -> None:
    lines = [
        f"dataset_id: {dataset_id}",
        "format: lerobot_v2.1",
        "robot: openarm",
        "task: Fold the T-shirt properly",
        f"created_at: {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}",
        f"source: {source}",
        f"source_split: {source_split}",
        "status: generated",
        "augmentation:",
        f"  original_episodes: {source_episodes}",
        f"  time_scaled_episodes: {time_episodes}",
        f"  mirrored_episodes: {mirror_episodes}",
        "  time_scaling:",
        "    extraction_factor: 2",
        "  space_mirroring:",
        "    vector_order: right_8d_left_8d_to_left_8d_right_8d",
        "    gripper: swap_only",
        f"episodes: {total_episodes}",
        f"frames: {total_frames}",
        "norm_stats: null",
        "video:",
        f"  ffmpeg: {ffmpeg}",
        f"  encoder: {encoder}",
        f"  workers: {video_workers}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _validate_source(info: dict[str, Any]) -> None:
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1 source, got {info.get('codebase_version')!r}")
    if "episode_{episode_index:06d}.parquet" not in info.get("data_path", ""):
        raise ValueError("Expected per-episode v2.1 parquet path in meta/info.json")
    for key in VECTOR_KEYS:
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [16]:
            raise ValueError(f"Expected {key} shape [16], got {shape}")
    for key in VIDEO_KEYS:
        if key not in info.get("features", {}):
            raise ValueError(f"Missing video feature {key}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=pathlib.Path,
        default=pathlib.Path("/share/home/linyongjia/datasets/high_quality_folding"),
    )
    parser.add_argument(
        "--dst", type=pathlib.Path, default=pathlib.Path("/share/home/linyongjia/datasets/openarm_hq_tda_aug_v1")
    )
    parser.add_argument("--dataset-id", default="openarm_hq_tda_aug_v1")
    parser.add_argument("--source-split", default="train", choices=("train", "val", "all"))
    parser.add_argument("--time-split-ratio", type=float, default=0.3)
    parser.add_argument("--extraction-factor", type=int, default=2)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--ffmpeg", type=pathlib.Path, default=None)
    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--video-quality", type=int, default=28)
    parser.add_argument("--video-preset", default="p4")
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--num-video-workers", type=int, default=4)
    parser.add_argument("--use-gpu-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-flip-wrist-videos", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--require-gpu-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug limit applied after split selection.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    info = _load_json(src / "meta/info.json")
    _validate_source(info)

    source_episode_ids = _parse_split(info, args.source_split)
    if args.max_episodes is not None:
        source_episode_ids = source_episode_ids[: args.max_episodes]
    time_count = math.ceil(len(source_episode_ids) * args.time_split_ratio)
    time_episode_ids = source_episode_ids[:time_count]

    total_episodes = len(source_episode_ids) + len(time_episode_ids) + len(source_episode_ids)
    print(f"source episodes: {len(source_episode_ids)} from split {args.source_split}")
    print(f"time-scaled episodes: {len(time_episode_ids)}")
    print(f"mirrored episodes: {len(source_episode_ids)}")
    print(f"target episodes: {total_episodes}")
    print(f"target: {dst}")
    if args.dry_run:
        return

    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            raise FileExistsError(f"{dst} exists and is not empty; pass --force to replace it")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    ffmpeg = args.ffmpeg
    if ffmpeg is None:
        ffmpeg = pathlib.Path(shutil.which("ffmpeg") or "/share/home/linyongjia/miniconda3/envs/pi-conda/bin/ffmpeg")
    source_episodes = {item["episode_index"]: item for item in _load_jsonl(src / "meta/episodes.jsonl")}
    source_stats = {item["episode_index"]: item["stats"] for item in _load_jsonl(src / "meta/episodes_stats.jsonl")}
    shutil.copy2(src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")

    target_episodes: list[dict[str, Any]] = []
    target_episode_stats: list[dict[str, Any]] = []
    video_jobs: list[dict[str, Any]] = []
    total_frames = 0
    video_link_counts = {"hardlink": 0, "copy": 0, "exists": 0}
    chunks_size = int(info["chunks_size"])
    fps = int(info["fps"])
    gpu_ids = [int(item) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    encoder = _select_encoder(
        ffmpeg,
        args.video_encoder,
        gpu_id=gpu_ids[0],
        preset=args.video_preset,
        quality=args.video_quality,
        require_gpu=args.require_gpu_video,
    )

    def append_episode(src_ep: int, operation: str, *, extraction_factor: int = 1, mirror: bool = False) -> None:
        nonlocal total_frames
        target_ep = len(target_episodes)
        src_data = src / _format_data_path(info, src_ep)
        dst_data = dst / _format_data_path(info, target_ep)

        can_link_original = (
            operation == "original"
            and target_ep == src_ep
            and total_frames == int(source_stats[src_ep]["index"]["min"][0])
        )
        if can_link_original:
            _copy_or_link(src_data, dst_data, args.copy_mode)
            length = int(source_episodes[src_ep]["length"])
            stats = source_stats[src_ep]
        else:
            episode_frame = _prepare_episode_dataframe(
                src_data,
                new_episode_index=target_ep,
                start_global_index=total_frames,
                fps=fps,
                extraction_factor=extraction_factor,
                mirror=mirror,
            )
            length = len(episode_frame)
            dst_data.parent.mkdir(parents=True, exist_ok=True)
            episode_frame.to_parquet(dst_data, index=False)
            stats = _episode_stats_for_transformed(episode_frame, source_stats[src_ep], mirror=mirror)

        target_episodes.append(
            {
                "episode_index": target_ep,
                "tasks": source_episodes[src_ep].get("tasks", []),
                "length": length,
                "source_dataset": str(src),
                "source_episode_index": src_ep,
                "augmentation_type": operation,
                "source_frame_stride": extraction_factor,
                "source_frame_offset": 0,
                "mirror": mirror,
            }
        )
        target_episode_stats.append({"episode_index": target_ep, "stats": stats})

        for video_key in VIDEO_KEYS:
            src_key = video_key
            hflip = False
            if operation == "mirror":
                if video_key == "observation.images.left_wrist":
                    src_key = "observation.images.right_wrist"
                    hflip = not args.no_flip_wrist_videos
                elif video_key == "observation.images.right_wrist":
                    src_key = "observation.images.left_wrist"
                    hflip = not args.no_flip_wrist_videos
                else:
                    hflip = True
            src_video = src / _format_video_path(info, src_ep, src_key)
            dst_video = dst / _format_video_path(info, target_ep, video_key)
            if operation == "original" and not args.skip_videos:
                mode = _copy_or_link(src_video, dst_video, args.copy_mode)
                video_link_counts[mode] = video_link_counts.get(mode, 0) + 1
            elif not args.skip_videos:
                video_jobs.append(
                    {
                        "src": str(src_video),
                        "dst": str(dst_video),
                        "ffmpeg": str(ffmpeg),
                        "encoder": encoder,
                        "gpu_id": gpu_ids[len(video_jobs) % len(gpu_ids)],
                        "fps": fps,
                        "extraction_factor": extraction_factor,
                        "hflip": hflip,
                        "use_gpu_decode": args.use_gpu_decode,
                        "quality": args.video_quality,
                        "preset": args.video_preset,
                    }
                )
        total_frames += length

    for src_ep in source_episode_ids:
        append_episode(src_ep, "original")
    for src_ep in time_episode_ids:
        append_episode(src_ep, "time", extraction_factor=args.extraction_factor)
    for src_ep in source_episode_ids:
        append_episode(src_ep, "mirror", mirror=True)

    target_info = dict(info)
    target_info["total_episodes"] = len(target_episodes)
    target_info["total_frames"] = total_frames
    target_info["total_tasks"] = info.get("total_tasks", 1)
    target_info["total_videos"] = len(target_episodes) * len(VIDEO_KEYS)
    target_info["total_chunks"] = math.ceil(len(target_episodes) / chunks_size)
    target_info["splits"] = {"train": f"0:{len(target_episodes)}"}

    _write_json(dst / "meta/info.json", target_info)
    _write_jsonl(dst / "meta/episodes.jsonl", target_episodes)
    _write_jsonl(dst / "meta/episodes_stats.jsonl", target_episode_stats)
    if (src / ".gitattributes").exists():
        shutil.copy2(src / ".gitattributes", dst / ".gitattributes")

    if video_jobs and not args.metadata_only:
        print(f"running {len(video_jobs)} ffmpeg video jobs with {args.num_video_workers} workers, encoder={encoder}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_video_workers) as executor:
            futures = [executor.submit(_run_video_job, job) for job in video_jobs]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                future.result()
                if idx % 50 == 0 or idx == len(futures):
                    print(f"video jobs complete: {idx}/{len(futures)}", flush=True)
    elif video_jobs:
        print(f"metadata-only: skipped {len(video_jobs)} ffmpeg video jobs")

    _write_manifest(
        dst / "manifest.yaml",
        dataset_id=args.dataset_id,
        source=src,
        source_split=args.source_split,
        source_episodes=len(source_episode_ids),
        time_episodes=len(time_episode_ids),
        mirror_episodes=len(source_episode_ids),
        total_episodes=len(target_episodes),
        total_frames=total_frames,
        ffmpeg=ffmpeg,
        encoder=encoder,
        video_workers=args.num_video_workers,
    )
    _write_json(
        dst / "augment_report.json",
        {
            "dataset_id": args.dataset_id,
            "source": str(src),
            "target": str(dst),
            "source_split": args.source_split,
            "source_episodes": len(source_episode_ids),
            "time_scaled_episodes": len(time_episode_ids),
            "mirrored_episodes": len(source_episode_ids),
            "time_extraction_factor": args.extraction_factor,
            "total_episodes": len(target_episodes),
            "total_frames": total_frames,
            "ffmpeg": str(ffmpeg),
            "encoder": encoder,
            "video_jobs": len(video_jobs),
            "video_link_counts": video_link_counts,
            "gpu_ids": gpu_ids,
        },
    )
    print(f"done: {dst}")
    print("next: recompute norm_stats.json before training")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
