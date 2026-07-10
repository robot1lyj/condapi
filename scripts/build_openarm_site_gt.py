"""Build Site-GT from complete OpenArm Site-A151 boundary annotations.

This is a deterministic data conversion. It does not load or run HQ-Stage (or
any other model). Source parquet and videos are left unchanged; output videos
are linked or copied into a separate LeRobot v2.1 dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import json
import os
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd
import tqdm

try:
    from scripts import openarm_stage_progress as stage_progress
    from scripts.write_lerobot_episode_stats import compute_episode_stats
except ModuleNotFoundError:
    import openarm_stage_progress as stage_progress
    from write_lerobot_episode_stats import compute_episode_stats


VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
ADVANTAGE_FEATURES = (
    "stage_progress_gt",
    "stage_id",
    "advantage_gt",
    "relative_advantage",
    "absolute_value",
    "absolute_advantage",
)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


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


def _copy_or_link(source: pathlib.Path, destination: pathlib.Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def build_gt_advantage(progress: np.ndarray, relative_interval: int) -> np.ndarray:
    """Compute the same interval-normalized delta used by HQ-Score."""

    if relative_interval <= 0:
        raise ValueError(f"relative_interval must be positive, got {relative_interval}")
    progress = np.asarray(progress, dtype=np.float32).reshape(-1)
    advantage = np.zeros(len(progress), dtype=np.float32)
    final_index = len(progress) - 1
    for frame_index in range(len(progress)):
        future_index = min(frame_index + relative_interval, final_index)
        delta = future_index - frame_index
        if delta == 0:
            continue
        scale = relative_interval / delta if delta != relative_interval else 1.0
        advantage[frame_index] = (progress[future_index] - progress[frame_index]) * scale
    return np.clip(advantage, -1.0, 1.0).astype(np.float32)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
    }


def _copy_support_files(source: pathlib.Path, destination: pathlib.Path) -> None:
    for child in source.iterdir():
        if child.name in {"data", "videos", "meta", "annotations"}:
            continue
        target = destination / child.name
        if child.is_file():
            shutil.copy2(child, target)
        elif child.is_dir():
            shutil.copytree(child, target)
    if (source / "annotations").exists():
        shutil.copytree(source / "annotations", destination / "annotations")


def build_site_gt_dataset(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    annotations_path: pathlib.Path | None,
    relative_interval: int,
    copy_mode: str,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    annotations_path = (annotations_path or source / "annotations/openarm_stage_v1.jsonl").resolve()
    info = _load_json(source / "meta/info.json")
    total_episodes = int(info["total_episodes"])
    episode_rows = {int(row["episode_index"]): row for row in _load_jsonl(source / "meta/episodes.jsonl")}
    annotations = {int(row["episode_index"]): row for row in _load_jsonl(annotations_path)}
    expected = set(range(total_episodes))
    if set(episode_rows) != expected:
        raise ValueError("Source episodes.jsonl must contain every contiguous episode index")
    if set(annotations) != expected:
        missing = sorted(expected - set(annotations))
        extra = sorted(set(annotations) - expected)
        raise ValueError(f"Site-A151 coverage mismatch: missing={missing}, extra={extra}")

    quality_counts = Counter(str(row.get("quality", "success")) for row in annotations.values())
    report: dict[str, Any] = {
        "source": str(source),
        "destination": str(destination),
        "annotations": str(annotations_path),
        "advantage_source": "site_gt",
        "relative_interval": relative_interval,
        "tail_normalization": "scale_short_final_windows_to_relative_interval",
        "total_episodes": total_episodes,
        "total_frames": int(info["total_frames"]),
        "quality_counts": dict(sorted(quality_counts.items())),
        "k_data_quality_filter": "success",
        "k_data_eligible_episodes": int(quality_counts.get("success", 0)),
        "copy_mode": copy_mode,
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    _copy_support_files(source, destination)

    output_episode_rows = []
    output_stats_rows = []
    all_advantages = []
    stage_counts: Counter[int] = Counter()
    for episode_index in tqdm.tqdm(range(total_episodes), desc="Building Site-GT"):
        source_parquet = source / _format_data_path(info, episode_index)
        frame = pd.read_parquet(source_parquet).copy()
        boundaries = stage_progress.normalize_boundaries(annotations[episode_index], len(frame))
        progress, stage_ids = stage_progress.build_stage_arrays(len(frame), boundaries)
        advantage = build_gt_advantage(progress, relative_interval)

        frame["stage_progress_gt"] = progress
        frame["stage_id"] = stage_ids
        frame["advantage_gt"] = advantage
        frame["relative_advantage"] = advantage
        frame["absolute_value"] = progress
        frame["absolute_advantage"] = advantage
        output_parquet = destination / _format_data_path(info, episode_index)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_parquet, index=False)

        row = dict(episode_rows[episode_index])
        annotation_quality = str(annotations[episode_index].get("quality", "success"))
        row["advantage_source"] = "site_gt"
        row["stage_annotation"] = "Site-A151"
        row["annotation_quality"] = annotation_quality
        row["eligible_for_k_data"] = annotation_quality == "success"
        row["flatten_done_frame"] = int(boundaries[0]["end_frame"])
        output_episode_rows.append(row)
        output_stats_rows.append({"episode_index": episode_index, "stats": compute_episode_stats(frame)})
        all_advantages.append(advantage)
        stage_counts.update(int(stage_id) for stage_id in stage_ids)

        for video_key in VIDEO_KEYS:
            source_video = source / _format_video_path(info, episode_index, video_key)
            destination_video = destination / _format_video_path(info, episode_index, video_key)
            _copy_or_link(source_video, destination_video, copy_mode)

    output_info = dict(info)
    output_info["repo_id"] = destination.name
    output_features = dict(output_info["features"])
    for feature in ADVANTAGE_FEATURES:
        dtype = "int64" if feature == "stage_id" else "float32"
        output_features[feature] = {"dtype": dtype, "shape": [1], "names": None}
    output_info["features"] = output_features
    output_info["site_gt"] = {
        "source_dataset": str(source),
        "annotations": str(destination / "annotations/openarm_stage_v1.jsonl"),
        "stage_annotation": "Site-A151",
        "advantage_source": "site_gt",
        "relative_interval": relative_interval,
        "hq_stage_used": False,
        "k_data_quality_filter": "success",
    }
    _write_json(destination / "meta/info.json", output_info)
    shutil.copy2(source / "meta/tasks.jsonl", destination / "meta/tasks.jsonl")
    _write_jsonl(destination / "meta/episodes.jsonl", output_episode_rows)
    _write_jsonl(destination / "meta/episodes_stats.jsonl", output_stats_rows)

    all_advantages_array = np.concatenate(all_advantages)
    report["advantage_summary"] = _summary(all_advantages_array)
    report["stage_frame_counts"] = {str(key): value for key, value in sorted(stage_counts.items())}
    report["output_features"] = list(ADVANTAGE_FEATURES)
    _write_json(destination / "site_gt_build_report.json", report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=pathlib.Path, help="Source Site LeRobot dataset")
    parser.add_argument("--destination", required=True, type=pathlib.Path, help="Destination Site-GT dataset")
    parser.add_argument("--annotations", type=pathlib.Path, help="Defaults to source Site-A151 JSONL")
    parser.add_argument("--relative-interval", type=int, default=50)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = build_site_gt_dataset(
        args.source,
        args.destination,
        annotations_path=args.annotations,
        relative_interval=args.relative_interval,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
