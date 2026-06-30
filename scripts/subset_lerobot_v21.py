"""Create a small, contiguous LeRobot v2.1 subset from an existing dataset.

This is intended for pulling only the OpenArm HQ v2.1 episodes needed for
Stage Advantage labeling. It rewrites episode indices to 0..N-1 and updates
parquet row metadata, while copying/linking the matching videos and meta files.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
import os
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_episodes(spec: str) -> list[int]:
    """Parse an episode spec like ``0:200`` or ``0,2,5``."""

    spec = spec.strip()
    if not spec:
        raise ValueError("Episode spec is empty")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid range episode spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, end, step))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


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


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]


def _copy_or_link(src: pathlib.Path, dst: pathlib.Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def _rewrite_parquet(src: pathlib.Path, dst: pathlib.Path, *, new_episode_index: int, start_index: int) -> int:
    episode_frame = pd.read_parquet(src).copy()
    length = len(episode_frame)
    episode_frame["episode_index"] = np.full(length, new_episode_index, dtype=np.int64)
    episode_frame["frame_index"] = np.arange(length, dtype=np.int64)
    episode_frame["index"] = np.arange(start_index, start_index + length, dtype=np.int64)
    if "timestamp" in episode_frame.columns and length:
        first_timestamp = float(episode_frame["timestamp"].iloc[0])
        if abs(first_timestamp) > 1e-6:
            episode_frame["timestamp"] = (
                episode_frame["timestamp"].astype(np.float32) - np.float32(first_timestamp)
            ).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    episode_frame.to_parquet(dst, index=False)
    return length


def create_subset(
    src: pathlib.Path,
    dst: pathlib.Path,
    episodes: list[int],
    *,
    copy_mode: str,
    overwrite: bool,
    val_count: int,
) -> dict[str, Any]:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    info = _load_json(src / "meta/info.json")
    episodes_meta = {int(row["episode_index"]): row for row in _load_jsonl(src / "meta/episodes.jsonl")}
    stats_path = src / "meta/episodes_stats.jsonl"
    stats_meta = {}
    if stats_path.exists():
        stats_meta = {int(row["episode_index"]): row for row in _load_jsonl(stats_path)}

    video_keys = _video_keys(info)
    chunks_size = int(info["chunks_size"])
    new_episode_rows: list[dict[str, Any]] = []
    new_stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    copied_videos = 0

    for new_episode_index, old_episode_index in enumerate(episodes):
        if old_episode_index not in episodes_meta:
            raise FileNotFoundError(f"Episode {old_episode_index} missing from meta/episodes.jsonl")

        old_data = src / _format_data_path(info, old_episode_index)
        new_data = dst / _format_data_path(info, new_episode_index)
        length = _rewrite_parquet(
            old_data,
            new_data,
            new_episode_index=new_episode_index,
            start_index=total_frames,
        )

        episode_row = dict(episodes_meta[old_episode_index])
        episode_row["episode_index"] = new_episode_index
        episode_row["length"] = length
        new_episode_rows.append(episode_row)

        if old_episode_index in stats_meta:
            stats_row = dict(stats_meta[old_episode_index])
            stats_row["episode_index"] = new_episode_index
            new_stats_rows.append(stats_row)

        for video_key in video_keys:
            old_video = src / _format_video_path(info, old_episode_index, video_key)
            new_video = dst / _format_video_path(info, new_episode_index, video_key)
            _copy_or_link(old_video, new_video, copy_mode)
            copied_videos += 1

        total_frames += length

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")
    new_info = dict(info)
    new_info["total_episodes"] = len(episodes)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = copied_videos
    new_info["total_chunks"] = max(1, math.ceil(len(episodes) / chunks_size))
    if val_count > 0:
        train_end = max(0, len(episodes) - val_count)
        new_info["splits"] = {"train": f"0:{train_end}", "val": f"{train_end}:{len(episodes)}"}
    else:
        new_info["splits"] = {"train": f"0:{len(episodes)}"}
    _write_json(dst / "meta/info.json", new_info)
    _write_jsonl(dst / "meta/episodes.jsonl", new_episode_rows)
    if new_stats_rows:
        _write_jsonl(dst / "meta/episodes_stats.jsonl", new_stats_rows)

    report = {
        "source": str(src),
        "destination": str(dst),
        "source_episodes": episodes,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_videos": copied_videos,
        "copy_mode": copy_mode,
    }
    _write_json(dst / "subset_report.json", report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=pathlib.Path, help="Source LeRobot v2.1 dataset root")
    parser.add_argument("--dst", required=True, type=pathlib.Path, help="Destination subset dataset root")
    parser.add_argument("--episodes", default="0:200", help="Episode spec, e.g. 0:200 or 0,2,5")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--val-count", type=int, default=0, help="Reserve the last N episodes as val split")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = create_subset(
        args.src,
        args.dst,
        parse_episodes(args.episodes),
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        val_count=args.val_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
