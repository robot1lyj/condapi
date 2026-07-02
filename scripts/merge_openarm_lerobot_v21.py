"""Merge OpenArm LeRobot v2.1 datasets into one trainable dataset.

The output is a normal single-repo LeRobot dataset. Sampling weights are
materialized by repeating selected source episodes with new episode indices.

Example:
    python scripts/merge_openarm_lerobot_v21.py \
      --dst /share/home/linyongjia/datasets/openarm_hq_tda_site_v1 \
      --source hq,/share/home/linyongjia/datasets/high_quality_folding,0:999,1 \
      --source tda,/share/home/linyongjia/datasets/openarm_hq_tda_aug_v1,0:2298,1 \
      --source site,/share/home/linyongjia/datasets/openarm_site_align_v1,0:130,5 \
      --copy-mode hardlink \
      --overwrite
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import dataclasses
import json
import math
import os
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    name: str
    path: pathlib.Path
    episodes: list[int]
    repeat: int

    @classmethod
    def parse(cls, raw: str) -> SourceSpec:
        parts = raw.split(",", 3)
        if len(parts) != 4:
            raise ValueError(
                "Source spec must be 'name,path,episodes,repeat', "
                f"for example 'site,/data/openarm_site_align_v1,0:130,5'; got {raw!r}"
            )
        name, path, episodes, repeat = parts
        repeat_int = int(repeat)
        if repeat_int <= 0:
            raise ValueError(f"Repeat must be positive for source {name!r}, got {repeat_int}")
        return cls(
            name=name,
            path=pathlib.Path(path),
            episodes=parse_episodes(episodes),
            repeat=repeat_int,
        )


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
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def _validate_compatible_info(base_info: dict[str, Any], source_info: dict[str, Any], source_name: str) -> None:
    for key in ("data_path", "video_path", "chunks_size", "fps"):
        if base_info.get(key) != source_info.get(key):
            raise ValueError(f"Source {source_name!r} has incompatible {key}: {source_info.get(key)!r}")

    required_features = (
        "observation.images.base",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.state",
        "action",
    )
    for feature_key in required_features:
        if feature_key not in source_info.get("features", {}):
            raise ValueError(f"Source {source_name!r} missing feature {feature_key!r}")
        base_feature = base_info["features"].get(feature_key, {})
        source_feature = source_info["features"].get(feature_key, {})
        if base_feature.get("shape") != source_feature.get("shape"):
            raise ValueError(
                f"Source {source_name!r} feature {feature_key!r} shape mismatch: "
                f"{source_feature.get('shape')} != {base_feature.get('shape')}"
            )


def _rewrite_parquet(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    new_episode_index: int,
    start_index: int,
    task_index: int,
) -> int:
    episode_frame = pd.read_parquet(src).copy()
    length = len(episode_frame)
    episode_frame["episode_index"] = np.full(length, new_episode_index, dtype=np.int64)
    episode_frame["frame_index"] = np.arange(length, dtype=np.int64)
    episode_frame["index"] = np.arange(start_index, start_index + length, dtype=np.int64)
    episode_frame["task_index"] = np.full(length, task_index, dtype=np.int64)
    if "timestamp" in episode_frame.columns and length:
        first_timestamp = float(episode_frame["timestamp"].iloc[0])
        if abs(first_timestamp) > 1e-6:
            episode_frame["timestamp"] = (
                episode_frame["timestamp"].astype(np.float32) - np.float32(first_timestamp)
            ).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    episode_frame.to_parquet(dst, index=False)
    return length


def merge_datasets(
    sources: list[SourceSpec],
    dst: pathlib.Path,
    *,
    task: str,
    copy_mode: str,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("At least one --source is required")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        if not dry_run:
            shutil.rmtree(dst)

    source_infos = {source.name: _load_json(source.path / "meta/info.json") for source in sources}
    base_info = source_infos[sources[0].name]
    for source in sources[1:]:
        _validate_compatible_info(base_info, source_infos[source.name], source.name)

    plan_rows = [
        {
            "name": source.name,
            "path": str(source.path),
            "episode_count": len(source.episodes),
            "repeat": source.repeat,
            "materialized_episodes": len(source.episodes) * source.repeat,
        }
        for source in sources
    ]
    planned_total = sum(row["materialized_episodes"] for row in plan_rows)
    if dry_run:
        report = {"destination": str(dst), "task": task, "planned_total_episodes": planned_total, "sources": plan_rows}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    dst.mkdir(parents=True)
    video_keys = _video_keys(base_info)
    chunks_size = int(base_info["chunks_size"])
    new_episode_rows: list[dict[str, Any]] = []
    new_stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    copied_videos = 0
    new_episode_index = 0

    for source in sources:
        info = source_infos[source.name]
        episodes_meta = {int(row["episode_index"]): row for row in _load_jsonl(source.path / "meta/episodes.jsonl")}
        stats_path = source.path / "meta/episodes_stats.jsonl"
        stats_meta = {}
        if stats_path.exists():
            stats_meta = {int(row["episode_index"]): row for row in _load_jsonl(stats_path)}

        for old_episode_index in source.episodes:
            if old_episode_index not in episodes_meta:
                raise FileNotFoundError(f"Episode {old_episode_index} missing from {source.path}/meta/episodes.jsonl")

            for repeat_index in range(source.repeat):
                del repeat_index
                old_data = source.path / _format_data_path(info, old_episode_index)
                new_data = dst / _format_data_path(base_info, new_episode_index)
                length = _rewrite_parquet(
                    old_data,
                    new_data,
                    new_episode_index=new_episode_index,
                    start_index=total_frames,
                    task_index=0,
                )

                episode_row = dict(episodes_meta[old_episode_index])
                episode_row["episode_index"] = new_episode_index
                episode_row["tasks"] = [task]
                episode_row["length"] = length
                episode_row["source_dataset"] = source.name
                episode_row["source_episode_index"] = old_episode_index
                new_episode_rows.append(episode_row)

                if old_episode_index in stats_meta:
                    stats_row = dict(stats_meta[old_episode_index])
                    stats_row["episode_index"] = new_episode_index
                    new_stats_rows.append(stats_row)

                for video_key in video_keys:
                    old_video = source.path / _format_video_path(info, old_episode_index, video_key)
                    new_video = dst / _format_video_path(base_info, new_episode_index, video_key)
                    _copy_or_link(old_video, new_video, copy_mode)
                    copied_videos += 1

                total_frames += length
                new_episode_index += 1

    new_info = dict(base_info)
    new_info["total_episodes"] = new_episode_index
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = copied_videos
    new_info["total_chunks"] = max(1, math.ceil(new_episode_index / chunks_size))
    new_info["splits"] = {"train": f"0:{new_episode_index}"}
    _write_json(dst / "meta/info.json", new_info)
    _write_jsonl(dst / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(dst / "meta/episodes.jsonl", new_episode_rows)
    if new_stats_rows:
        _write_jsonl(dst / "meta/episodes_stats.jsonl", new_stats_rows)

    report = {
        "destination": str(dst),
        "task": task,
        "total_episodes": new_episode_index,
        "total_frames": total_frames,
        "total_videos": copied_videos,
        "copy_mode": copy_mode,
        "sources": plan_rows,
    }
    _write_json(dst / "merge_report.json", report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="Destination LeRobot dataset root")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source spec: name,path,episodes,repeat. Example: site,/data/openarm_site_align_v1,0:130,5",
    )
    parser.add_argument("--task", default="fold the cloth")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = merge_datasets(
        [SourceSpec.parse(raw) for raw in args.source],
        args.dst,
        task=args.task,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
