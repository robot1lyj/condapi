"""Apply OpenArm Stage Advantage annotations to a LeRobot v2.1 dataset.

The script reads sidecar JSONL annotations produced by
``openarm_stage_annotator.py`` and writes per-frame ``stage_progress_gt`` plus
``stage_id`` columns into episode parquet files. For Task A alignment, the
default schema is two stages: flattening and folding.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

STAGES = ("flattening", "folding")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_data_path(info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(info["data_path"].format(episode_chunk=chunk, episode_index=episode_index))


def _event_frame(annotation: dict[str, Any], *names: str) -> int | None:
    for event in annotation.get("events", []):
        if event.get("name") in names:
            return int(event["frame"])
    for name in names:
        if name in annotation and annotation[name] is not None:
            return int(annotation[name])
    return None


def normalize_boundaries(annotation: dict[str, Any], episode_length: int) -> list[dict[str, int | str]]:
    """Return two Task-A boundaries from sidecar annotation data."""

    if annotation.get("stage_boundaries"):
        boundaries = [
            {
                "stage_id": int(boundary["stage_id"]),
                "name": str(boundary.get("name", STAGES[int(boundary["stage_id"])])),
                "start_frame": int(boundary["start_frame"]),
                "end_frame": int(boundary["end_frame"]),
            }
            for boundary in annotation["stage_boundaries"]
        ]
        boundaries.sort(key=lambda item: int(item["stage_id"]))
        return boundaries

    start = _event_frame(annotation, "episode_start", "start")
    flatten_done = _event_frame(annotation, "flatten_done")
    fold_start = _event_frame(annotation, "fold_start")
    end = _event_frame(annotation, "episode_end", "end")
    if start is None:
        start = 0
    if end is None:
        end = episode_length - 1
    if flatten_done is None and fold_start is None:
        raise ValueError(f"Episode {annotation.get('episode_index')} missing flatten_done/fold_start")
    if fold_start is None:
        fold_start = min(int(flatten_done) + 1, end)
    if flatten_done is None:
        flatten_done = max(start, int(fold_start) - 1)

    return [
        {"stage_id": 0, "name": "flattening", "start_frame": start, "end_frame": int(fold_start) - 1},
        {"stage_id": 1, "name": "folding", "start_frame": fold_start, "end_frame": end},
    ]


def validate_boundaries(boundaries: list[dict[str, Any]], episode_length: int) -> None:
    if len(boundaries) != len(STAGES):
        raise ValueError(f"Expected {len(STAGES)} boundaries, got {len(boundaries)}")
    previous_start = -1
    previous_end = -1
    for expected_stage_id, boundary in enumerate(boundaries):
        stage_id = int(boundary["stage_id"])
        start = int(boundary["start_frame"])
        end = int(boundary["end_frame"])
        if stage_id != expected_stage_id:
            raise ValueError(f"Expected stage_id={expected_stage_id}, got {stage_id}")
        if start < 0 or end < 0 or start >= episode_length or end >= episode_length:
            raise ValueError(f"Boundary out of range for length {episode_length}: {boundary}")
        if start > end:
            raise ValueError(f"Boundary start > end: {boundary}")
        if start < previous_start or end < previous_end:
            raise ValueError(f"Boundaries must be monotonic: {boundaries}")
        if expected_stage_id > 0 and start != previous_end + 1:
            raise ValueError(f"Boundaries must be adjacent without overlap or gaps: {boundaries}")
        previous_start = start
        previous_end = end


def build_stage_arrays(
    episode_length: int,
    boundaries: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``stage_progress_gt`` and ``stage_id`` arrays for one episode."""

    validate_boundaries(boundaries, episode_length)
    progress = np.zeros(episode_length, dtype=np.float32)
    stage_ids = np.zeros(episode_length, dtype=np.int64)
    stage_count = len(boundaries)

    for boundary in boundaries:
        stage_id = int(boundary["stage_id"])
        start = int(boundary["start_frame"])
        end = int(boundary["end_frame"])
        span = max(1, end - start)
        frames = np.arange(start, end + 1, dtype=np.float32)
        local = (frames - float(start)) / float(span)
        progress[start : end + 1] = (float(stage_id) + local) / float(stage_count)
        stage_ids[start : end + 1] = stage_id

    first_start = int(boundaries[0]["start_frame"])
    final_end = int(boundaries[-1]["end_frame"])
    progress[:first_start] = 0.0
    stage_ids[:first_start] = 0
    progress[final_end + 1 :] = 1.0
    stage_ids[final_end + 1 :] = stage_count - 1
    return progress, stage_ids


def _ensure_info_features(dataset: pathlib.Path) -> None:
    info_path = dataset / "meta/info.json"
    info = _load_json(info_path)
    features = dict(info["features"])
    features["stage_progress_gt"] = {"dtype": "float32", "shape": [1], "names": None}
    features["stage_id"] = {"dtype": "int64", "shape": [1], "names": None}
    info["features"] = features
    _write_json(info_path, info)


def _annotation_map(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    annotations: dict[int, dict[str, Any]] = {}
    for row in rows:
        annotations[int(row["episode_index"])] = row
    return annotations


def apply_annotations(
    dataset: pathlib.Path,
    annotations_path: pathlib.Path,
    *,
    dry_run: bool,
    quality_filter: set[str] | None,
) -> dict[str, Any]:
    info = _load_json(dataset / "meta/info.json")
    annotations = _annotation_map(_load_jsonl(annotations_path))
    if not annotations:
        raise ValueError(f"No annotations found in {annotations_path}")

    processed = []
    skipped = []
    for episode_index, annotation in sorted(annotations.items()):
        quality = str(annotation.get("quality", "success"))
        if quality_filter is not None and quality not in quality_filter:
            skipped.append({"episode_index": episode_index, "reason": f"quality={quality}"})
            continue

        parquet_path = dataset / _format_data_path(info, episode_index)
        episode_frame = pd.read_parquet(parquet_path)
        boundaries = normalize_boundaries(annotation, len(episode_frame))
        progress, stage_ids = build_stage_arrays(len(episode_frame), boundaries)
        monotonic = bool(np.all(np.diff(progress) >= -1e-6))
        if not monotonic:
            raise ValueError(f"stage_progress_gt is not monotonic for episode {episode_index}")

        if not dry_run:
            episode_frame = episode_frame.copy()
            episode_frame["stage_progress_gt"] = progress
            episode_frame["stage_id"] = stage_ids
            episode_frame.to_parquet(parquet_path, index=False)

        processed.append(
            {
                "episode_index": episode_index,
                "quality": quality,
                "length": len(episode_frame),
                "stage_progress_min": float(progress.min()),
                "stage_progress_max": float(progress.max()),
                "boundaries": boundaries,
            }
        )

    if not dry_run:
        _ensure_info_features(dataset)

    report = {
        "dataset": str(dataset),
        "annotations": str(annotations_path),
        "dry_run": dry_run,
        "processed_count": len(processed),
        "skipped_count": len(skipped),
        "processed": processed,
        "skipped": skipped,
    }
    if not dry_run:
        _write_json(dataset / "annotations/stage_progress_report.json", report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=pathlib.Path, help="LeRobot v2.1 dataset root")
    parser.add_argument(
        "--annotations",
        type=pathlib.Path,
        help="Sidecar JSONL path; defaults to <dataset>/annotations/openarm_stage_v1.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing parquet files")
    parser.add_argument(
        "--quality",
        default="success",
        help="Comma-separated quality values to apply. Use 'all' to include every annotation.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    annotations = args.annotations or args.dataset / "annotations/openarm_stage_v1.jsonl"
    quality_filter = (
        None if args.quality == "all" else {part.strip() for part in args.quality.split(",") if part.strip()}
    )
    report = apply_annotations(args.dataset, annotations, dry_run=args.dry_run, quality_filter=quality_filter)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
