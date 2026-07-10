"""Build a labeled, source-preserving Site Stage dataset from manual two-stage annotations."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.openarm_stage_progress import build_stage_arrays
    from scripts.openarm_stage_progress import normalize_boundaries
except ModuleNotFoundError:
    from openarm_stage_progress import build_stage_arrays
    from openarm_stage_progress import normalize_boundaries


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json_atomic(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


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


def _link(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_parquet_atomic(frame: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_site_stage_dataset(
    source: pathlib.Path,
    annotations_path: pathlib.Path,
    destination: pathlib.Path,
    *,
    validation_start: int,
    expected_train: int | None,
    expected_validation: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    info = _load_json(source / "meta/info.json")
    episode_meta = {int(row["episode_index"]): row for row in _load_jsonl(source / "meta/episodes.jsonl")}
    annotations = {int(row["episode_index"]): row for row in _load_jsonl(annotations_path)}
    success = sorted(
        episode_index
        for episode_index, annotation in annotations.items()
        if str(annotation.get("quality", "success")) == "success"
    )
    train_sources = [episode_index for episode_index in success if episode_index < validation_start]
    validation_sources = [episode_index for episode_index in success if episode_index >= validation_start]
    if expected_train is not None and len(train_sources) != expected_train:
        raise ValueError(f"Expected {expected_train} train Site episodes, found {len(train_sources)}")
    if expected_validation is not None and len(validation_sources) != expected_validation:
        raise ValueError(f"Expected {expected_validation} validation Site episodes, found {len(validation_sources)}")

    ordered_sources = train_sources + validation_sources
    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    output_episode_rows = []
    output_annotation_rows = []
    total_frames = 0
    total_videos = 0

    for new_episode_index, source_episode_index in enumerate(ordered_sources):
        source_parquet = source / _format_data_path(info, source_episode_index)
        output_parquet = destination / _format_data_path(info, new_episode_index)
        frame = pd.read_parquet(source_parquet).copy()
        boundaries = normalize_boundaries(annotations[source_episode_index], len(frame))
        progress, stage_ids = build_stage_arrays(len(frame), boundaries)
        frame["episode_index"] = np.full(len(frame), new_episode_index, dtype=np.int64)
        frame["frame_index"] = np.arange(len(frame), dtype=np.int64)
        frame["index"] = np.arange(total_frames, total_frames + len(frame), dtype=np.int64)
        frame["task_index"] = np.zeros(len(frame), dtype=np.int64)
        frame["stage_progress_gt"] = progress
        frame["stage_id"] = stage_ids
        _write_parquet_atomic(frame, output_parquet)

        episode_row = dict(episode_meta[source_episode_index])
        episode_row["episode_index"] = new_episode_index
        episode_row["source_dataset"] = "Site-A150"
        episode_row["source_episode_index"] = source_episode_index
        episode_row["tasks"] = ["Fold the T-shirt properly"]
        episode_row["length"] = len(frame)
        episode_row["split"] = "train" if new_episode_index < len(train_sources) else "validation"
        output_episode_rows.append(episode_row)

        annotation_row = dict(annotations[source_episode_index])
        annotation_row["episode_index"] = new_episode_index
        annotation_row["source_episode_index"] = source_episode_index
        output_annotation_rows.append(annotation_row)

        for video_key in video_keys:
            source_video = source / _format_video_path(info, source_episode_index, video_key)
            output_video = destination / _format_video_path(info, new_episode_index, video_key)
            _link(source_video, output_video)
            total_videos += 1
        total_frames += len(frame)

    output_info = dict(info)
    output_features = dict(output_info["features"])
    output_features["stage_progress_gt"] = {"dtype": "float32", "shape": [1], "names": None}
    output_features["stage_id"] = {"dtype": "int64", "shape": [1], "names": None}
    output_info["features"] = output_features
    output_info["total_episodes"] = len(ordered_sources)
    output_info["total_frames"] = total_frames
    output_info["total_videos"] = total_videos
    output_info["total_tasks"] = 1
    output_info["total_chunks"] = max(1, math.ceil(len(ordered_sources) / int(info["chunks_size"])))
    output_info["splits"] = {
        "train": f"0:{len(train_sources)}",
        "validation": f"{len(train_sources)}:{len(ordered_sources)}",
    }
    _write_json_atomic(destination / "meta/info.json", output_info)
    _write_jsonl_atomic(
        destination / "meta/tasks.jsonl",
        [{"task_index": 0, "task": "Fold the T-shirt properly"}],
    )
    _write_jsonl_atomic(destination / "meta/episodes.jsonl", output_episode_rows)
    _write_jsonl_atomic(destination / "annotations/openarm_stage_v1.jsonl", output_annotation_rows)

    report = {
        "schema_version": "openarm_site_stage_dataset_v1",
        "source": str(source),
        "annotations": str(annotations_path),
        "destination": str(destination),
        "train_episodes": len(train_sources),
        "validation_episodes": len(validation_sources),
        "total_episodes": len(ordered_sources),
        "total_frames": total_frames,
        "source_train_episode_indices": train_sources,
        "source_validation_episode_indices": validation_sources,
    }
    _write_json_atomic(destination / "stage_dataset_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--annotations", required=True, type=pathlib.Path)
    parser.add_argument("--destination", required=True, type=pathlib.Path)
    parser.add_argument("--validation-start", type=int, default=141)
    parser.add_argument("--expected-train", type=int, default=140)
    parser.add_argument("--expected-validation", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_site_stage_dataset(
        args.source,
        args.annotations,
        args.destination,
        validation_start=args.validation_start,
        expected_train=args.expected_train,
        expected_validation=args.expected_validation,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
