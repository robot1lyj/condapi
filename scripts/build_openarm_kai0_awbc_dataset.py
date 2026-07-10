"""Build the formal binary OpenArm KAI0 AWBC dataset from HQ, Site, and small-budget TDA scores."""

from __future__ import annotations

import argparse
from collections import Counter
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

try:
    from scripts.openarm_kai0_contract import HQ_FOLDING_ONLY_START
    from scripts.openarm_stage_progress import build_stage_arrays
    from scripts.openarm_stage_progress import normalize_boundaries
except ImportError:
    from openarm_kai0_contract import HQ_FOLDING_ONLY_START
    from openarm_stage_progress import build_stage_arrays
    from openarm_stage_progress import normalize_boundaries

VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
SCORE_COLUMNS = ("relative_advantage", "absolute_value", "absolute_advantage")
TASK = "Fold the T-shirt properly"
TASKS = (
    {"task_index": 0, "task": f"{TASK}, Advantage: negative"},
    {"task_index": 1, "task": f"{TASK}, Advantage: positive"},
)


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    kind: str
    dataset: pathlib.Path
    info: dict[str, Any]
    episode_index: int
    source_episode_index: int
    metadata: dict[str, Any]
    scores: dict[str, np.ndarray] | None = None
    stage_ids: np.ndarray | None = None


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


def _write_jsonl_atomic(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _write_parquet_atomic(frame: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _read_scores(source: EpisodeSource) -> dict[str, np.ndarray]:
    if source.scores is not None:
        return source.scores
    frame = pd.read_parquet(
        source.dataset / _format_data_path(source.info, source.episode_index),
        columns=list(SCORE_COLUMNS),
    )
    result = {column: frame[column].to_numpy(dtype=np.float32) for column in SCORE_COLUMNS}
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError(f"Non-finite score in {source.kind} source episode {source.source_episode_index}")
    return result


def _predicted_stage_ids(scores: dict[str, np.ndarray]) -> np.ndarray:
    return (scores["absolute_value"] >= 0.5).astype(np.int64)


def _hq_stage_ids(source: EpisodeSource, scores: dict[str, np.ndarray], folding_only_start: int) -> np.ndarray:
    if source.source_episode_index >= folding_only_start:
        return np.ones(len(scores["absolute_value"]), dtype=np.int64)
    return _predicted_stage_ids(scores)


def _site_stage_ids(
    source: EpisodeSource,
    scores: dict[str, np.ndarray],
    annotations: dict[int, dict[str, Any]] | None,
) -> np.ndarray:
    if annotations is None:
        return _predicted_stage_ids(scores)
    annotation = annotations.get(source.source_episode_index)
    if annotation is None:
        raise ValueError(f"Missing Site stage annotation for source episode {source.source_episode_index}")
    boundaries = normalize_boundaries(annotation, len(scores["absolute_value"]))
    _, stage_ids = build_stage_arrays(len(scores["absolute_value"]), boundaries)
    return stage_ids


def _read_stage_ids(
    source: EpisodeSource,
    scores: dict[str, np.ndarray],
    *,
    hq_folding_only_start: int,
    site_annotations: dict[int, dict[str, Any]] | None,
) -> np.ndarray:
    if source.stage_ids is not None:
        return source.stage_ids
    if source.kind == "HQ":
        return _hq_stage_ids(source, scores, hq_folding_only_start)
    if source.kind == "Site":
        return _site_stage_ids(source, scores, site_annotations)
    raise ValueError(f"Missing explicit stage IDs for source kind {source.kind}")


def _index_score_roots(roots: list[pathlib.Path], kind: str) -> dict[int, EpisodeSource]:
    indexed: dict[int, EpisodeSource] = {}
    for root_path in roots:
        root = root_path.resolve()
        info = _load_json(root / "meta/info.json")
        for row in _load_jsonl(root / "meta/episodes.jsonl"):
            local_episode = int(row["episode_index"])
            source_episode = int(row["source_episode_index"])
            if source_episode in indexed:
                raise ValueError(f"Duplicate {kind} source episode {source_episode}")
            indexed[source_episode] = EpisodeSource(
                kind=kind,
                dataset=root,
                info=info,
                episode_index=local_episode,
                source_episode_index=source_episode,
                metadata=row,
            )
    return indexed


def _validate_source_episode_ids(indexed: dict[int, EpisodeSource], expected: set[int], kind: str) -> None:
    actual = set(indexed)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise ValueError(
        f"{kind} score episode IDs do not match the expected split: "
        f"missing={missing[:20]}, unexpected={unexpected[:20]}"
    )


def _evenly_select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 0 or count > len(rows):
        raise ValueError(f"Cannot select {count} rows from {len(rows)}")
    if count == 0:
        return []
    positions = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[int(position)] for position in positions]


def _map_tda_scores(
    source_scores: dict[str, np.ndarray],
    *,
    target_length: int,
    stride: int,
    offset: int,
    relative_interval: int,
) -> dict[str, np.ndarray]:
    source_length = len(source_scores["absolute_value"])
    source_indices = np.minimum(offset + np.arange(target_length) * stride, source_length - 1).astype(np.int64)
    absolute_value = source_scores["absolute_value"][source_indices].astype(np.float32)
    relative = np.zeros(target_length, dtype=np.float32)
    absolute = np.zeros(target_length, dtype=np.float32)

    for frame_index in range(target_length):
        future_index = min(frame_index + relative_interval, target_length - 1)
        if future_index == frame_index:
            continue
        delta = future_index - frame_index
        start_source = int(source_indices[frame_index])
        end_source = int(source_indices[future_index])
        cursor = start_source
        direct_sum = 0.0
        while cursor < end_source:
            segment = min(relative_interval, end_source - cursor)
            direct_sum += float(source_scores["relative_advantage"][cursor]) * (segment / relative_interval)
            cursor += segment
        tail_scale = relative_interval / delta if delta != relative_interval else 1.0
        relative[frame_index] = direct_sum * tail_scale
        absolute[frame_index] = (absolute_value[future_index] - absolute_value[frame_index]) * tail_scale

    return {
        "relative_advantage": np.clip(relative, -1.0, 1.0).astype(np.float32),
        "absolute_value": np.clip(absolute_value, -1.0, 1.0).astype(np.float32),
        "absolute_advantage": np.clip(absolute, -1.0, 1.0).astype(np.float32),
    }


def _build_tda_sources(
    augmented: pathlib.Path,
    hq_scores: dict[int, EpisodeSource],
    *,
    time_count: int,
    mirror_count: int,
    relative_interval: int,
    hq_folding_only_start: int,
) -> list[EpisodeSource]:
    augmented = augmented.resolve()
    info = _load_json(augmented / "meta/info.json")
    rows = _load_jsonl(augmented / "meta/episodes.jsonl")
    time_rows = _evenly_select([row for row in rows if row.get("augmentation_type") == "time"], time_count)
    mirror_rows = _evenly_select([row for row in rows if row.get("augmentation_type") == "mirror"], mirror_count)
    selected = time_rows + mirror_rows
    result = []
    for row in selected:
        local_episode = int(row["episode_index"])
        source_episode = int(row["source_episode_index"])
        if source_episode not in hq_scores:
            raise ValueError(f"TDA episode maps to missing HQ score {source_episode}")
        target_frame = pd.read_parquet(
            augmented / _format_data_path(info, local_episode),
            columns=["frame_index"],
        )
        source_score = _read_scores(hq_scores[source_episode])
        mapped = _map_tda_scores(
            source_score,
            target_length=len(target_frame),
            stride=int(row.get("source_frame_stride", 1)),
            offset=int(row.get("source_frame_offset", 0)),
            relative_interval=relative_interval,
        )
        source_indices = np.minimum(
            int(row.get("source_frame_offset", 0))
            + np.arange(len(target_frame), dtype=np.int64) * int(row.get("source_frame_stride", 1)),
            len(source_score["absolute_value"]) - 1,
        )
        mapped_stage_ids = _hq_stage_ids(hq_scores[source_episode], source_score, hq_folding_only_start)[source_indices]
        result.append(
            EpisodeSource(
                kind="TDA",
                dataset=augmented,
                info=info,
                episode_index=local_episode,
                source_episode_index=source_episode,
                metadata=row,
                scores=mapped,
                stage_ids=mapped_stage_ids,
            )
        )
    return result


def _validate_compatible(base: dict[str, Any], other: dict[str, Any], kind: str) -> None:
    for key in ("fps", "chunks_size", "data_path", "video_path"):
        if base.get(key) != other.get(key):
            raise ValueError(f"{kind} has incompatible {key}: {other.get(key)!r} != {base.get(key)!r}")
    for key in (*VIDEO_KEYS, "observation.state", "action"):
        if base["features"][key].get("shape") != other["features"][key].get("shape"):
            raise ValueError(f"{kind} has incompatible feature shape for {key}")


def _serialize_counts(counts: dict[str, dict[int, Counter]]) -> dict[str, Any]:
    result = {}
    for kind, stages in counts.items():
        result[kind] = {}
        for stage_index, counter in stages.items():
            total = sum(counter.values())
            positive = int(counter[1])
            result[kind][str(stage_index)] = {
                "negative": int(counter[0]),
                "positive": positive,
                "total": total,
                "positive_ratio": positive / total if total else 0.0,
            }
    return result


def _audit_label_ratios(
    counts: dict[str, Any],
    *,
    positive_ratio: float,
    ratio_tolerance: float,
    source_ratio_bounds: tuple[float, float],
) -> dict[str, Any]:
    overall = {}
    source_ratio_violations = []
    for stage_index in (0, 1):
        stage_key = str(stage_index)
        negative = sum(int(counts[kind][stage_key]["negative"]) for kind in counts)
        positive = sum(int(counts[kind][stage_key]["positive"]) for kind in counts)
        if positive + negative == 0:
            raise ValueError(f"Stage {stage_index} has no frames")
        ratio = positive / (positive + negative)
        overall[stage_key] = {"negative": negative, "positive": positive, "positive_ratio": ratio}
        lower = positive_ratio - ratio_tolerance
        upper = positive_ratio + ratio_tolerance
        if not lower <= ratio <= upper:
            raise ValueError(f"Stage {stage_index} positive ratio {ratio:.4f} is outside [{lower:.4f}, {upper:.4f}]")
        for kind in counts:
            if int(counts[kind][stage_key]["total"]) == 0:
                continue
            source_ratio = float(counts[kind][stage_key]["positive_ratio"])
            if not source_ratio_bounds[0] <= source_ratio <= source_ratio_bounds[1]:
                source_ratio_violations.append(
                    {
                        "source_kind": kind,
                        "stage_index": stage_index,
                        "positive_ratio": source_ratio,
                    }
                )
    if source_ratio_violations:
        raise ValueError(
            "Per-source positive ratios are outside "
            f"{source_ratio_bounds}: {json.dumps(source_ratio_violations, ensure_ascii=False)}"
        )
    return overall


def build_kai0_awbc_dataset(
    hq_score_roots: list[pathlib.Path],
    site_score_roots: list[pathlib.Path],
    tda_augmented: pathlib.Path,
    destination: pathlib.Path,
    *,
    site_repeat: int,
    tda_time_count: int,
    tda_mirror_count: int,
    positive_ratio: float,
    relative_interval: int,
    overwrite: bool,
    expected_hq_episodes: int = 999,
    site_train_source_end: int = 141,
    excluded_site_source_episodes: tuple[int, ...] = (95,),
    ratio_tolerance: float = 0.01,
    source_ratio_bounds: tuple[float, float] = (0.15, 0.45),
    hq_folding_only_start: int = HQ_FOLDING_ONLY_START,
    site_annotations_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    if not 0.0 < positive_ratio < 1.0:
        raise ValueError("positive_ratio must be in (0, 1)")
    if site_repeat <= 0:
        raise ValueError("site_repeat must be positive")
    if relative_interval <= 0:
        raise ValueError("relative_interval must be positive")
    if expected_hq_episodes <= 0 or site_train_source_end <= 0:
        raise ValueError("Expected episode counts must be positive")
    if not 0 < hq_folding_only_start <= expected_hq_episodes:
        raise ValueError("hq_folding_only_start must lie within the HQ episode range")
    if ratio_tolerance < 0.0:
        raise ValueError("ratio_tolerance must be non-negative")
    if not 0.0 <= source_ratio_bounds[0] <= source_ratio_bounds[1] <= 1.0:
        raise ValueError("source_ratio_bounds must be within [0, 1]")

    destination = destination.resolve()
    input_roots = {path.resolve() for path in (*hq_score_roots, *site_score_roots, tda_augmented)}
    if destination in input_roots:
        raise ValueError("Destination must not overwrite an input dataset")

    hq_scores = _index_score_roots(hq_score_roots, "HQ")
    site_scores_all = _index_score_roots(site_score_roots, "Site")
    _validate_source_episode_ids(hq_scores, set(range(expected_hq_episodes)), "HQ")
    expected_site_ids = set(range(site_train_source_end)) - set(excluded_site_source_episodes)
    site_scores = {episode: source for episode, source in site_scores_all.items() if episode in expected_site_ids}
    _validate_source_episode_ids(site_scores, expected_site_ids, "Site train")
    site_annotations = None
    if site_annotations_path is not None:
        site_annotations = {int(row["episode_index"]): row for row in _load_jsonl(site_annotations_path.resolve())}
        missing_site_annotations = sorted(expected_site_ids - set(site_annotations))
        if missing_site_annotations:
            raise ValueError(f"Missing Site annotations: {missing_site_annotations[:20]}")

    tda_sources = _build_tda_sources(
        tda_augmented,
        hq_scores,
        time_count=tda_time_count,
        mirror_count=tda_mirror_count,
        relative_interval=relative_interval,
        hq_folding_only_start=hq_folding_only_start,
    )
    unique_sources = [*hq_scores.values(), *site_scores.values(), *tda_sources]
    base_info = unique_sources[0].info
    for source in unique_sources[1:]:
        _validate_compatible(base_info, source.info, source.kind)

    values_by_stage = {0: [], 1: []}
    for source in unique_sources:
        scores = _read_scores(source)
        stages = _read_stage_ids(
            source,
            scores,
            hq_folding_only_start=hq_folding_only_start,
            site_annotations=site_annotations,
        )
        for stage_index in (0, 1):
            values_by_stage[stage_index].append(scores["relative_advantage"][stages == stage_index])
    stage_values = {stage_index: np.concatenate(values) for stage_index, values in values_by_stage.items()}
    if any(len(values) == 0 for values in stage_values.values()):
        raise ValueError("Both predicted stages must contain frames before discretization")
    thresholds = {
        stage_index: float(np.percentile(values, (1.0 - positive_ratio) * 100.0))
        for stage_index, values in stage_values.items()
    }

    unique_label_counts = {kind: {0: Counter(), 1: Counter()} for kind in ("HQ", "Site", "TDA")}
    for source in unique_sources:
        scores = _read_scores(source)
        stages = _read_stage_ids(
            source,
            scores,
            hq_folding_only_start=hq_folding_only_start,
            site_annotations=site_annotations,
        )
        labels = np.asarray(
            [
                int(value >= thresholds[int(stage)])
                for value, stage in zip(scores["relative_advantage"], stages, strict=True)
            ],
            dtype=np.int64,
        )
        for stage_index in (0, 1):
            unique_label_counts[source.kind][stage_index].update(labels[stages == stage_index].tolist())
    unique_counts = _serialize_counts(unique_label_counts)
    overall_unique = _audit_label_ratios(
        unique_counts,
        positive_ratio=positive_ratio,
        ratio_tolerance=ratio_tolerance,
        source_ratio_bounds=source_ratio_bounds,
    )

    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
        shutil.rmtree(destination)
    staging_destination = destination.with_name(f".{destination.name}.building-{os.getpid()}")
    shutil.rmtree(staging_destination, ignore_errors=True)
    staging_destination.mkdir(parents=True)

    output_episode_rows = []
    materialized_label_counts = {kind: {0: Counter(), 1: Counter()} for kind in ("HQ", "Site", "TDA")}
    total_frames = 0
    total_videos = 0
    new_episode_index = 0

    for source in unique_sources:
        raw_frame = pd.read_parquet(source.dataset / _format_data_path(source.info, source.episode_index))
        scores = _read_scores(source)
        if len(raw_frame) != len(scores["relative_advantage"]):
            raise ValueError(f"Score length mismatch for {source.kind} episode {source.source_episode_index}")
        stages = _read_stage_ids(
            source,
            scores,
            hq_folding_only_start=hq_folding_only_start,
            site_annotations=site_annotations,
        )
        labels = np.asarray(
            [
                int(value >= thresholds[int(stage)])
                for value, stage in zip(scores["relative_advantage"], stages, strict=True)
            ],
            dtype=np.int64,
        )
        repeat = site_repeat if source.kind == "Site" else 1
        for repeat_index in range(repeat):
            frame = raw_frame.copy()
            length = len(frame)
            frame["episode_index"] = np.full(length, new_episode_index, dtype=np.int64)
            frame["frame_index"] = np.arange(length, dtype=np.int64)
            frame["index"] = np.arange(total_frames, total_frames + length, dtype=np.int64)
            frame["task_index"] = labels
            for column in SCORE_COLUMNS:
                frame[column] = scores[column]
            frame["stage_id_awbc"] = stages
            output_parquet = staging_destination / _format_data_path(base_info, new_episode_index)
            _write_parquet_atomic(frame, output_parquet)

            episode_row = dict(source.metadata)
            episode_row.update(
                {
                    "episode_index": new_episode_index,
                    "length": length,
                    "tasks": [row["task"] for row in TASKS],
                    "source_kind": source.kind,
                    "source_episode_index": source.source_episode_index,
                    "source_local_episode_index": source.episode_index,
                    "repeat_index": repeat_index,
                    "score_origin": "Stage model prediction",
                }
            )
            output_episode_rows.append(episode_row)
            for stage_index in (0, 1):
                materialized_label_counts[source.kind][stage_index].update(labels[stages == stage_index].tolist())

            for video_key in VIDEO_KEYS:
                source_video = source.dataset / _format_video_path(source.info, source.episode_index, video_key)
                output_video = staging_destination / _format_video_path(base_info, new_episode_index, video_key)
                _link(source_video, output_video)
                total_videos += 1
            total_frames += length
            new_episode_index += 1

    expected_episodes = expected_hq_episodes + len(expected_site_ids) * site_repeat + tda_time_count + tda_mirror_count
    if new_episode_index != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} materialized episodes, wrote {new_episode_index}")

    materialized_counts = _serialize_counts(materialized_label_counts)

    output_info = dict(base_info)
    features = dict(output_info["features"])
    for column in SCORE_COLUMNS:
        features[column] = {"dtype": "float32", "shape": [1], "names": None}
    features["stage_id_awbc"] = {"dtype": "int64", "shape": [1], "names": None}
    output_info["features"] = features
    output_info["total_episodes"] = new_episode_index
    output_info["total_frames"] = total_frames
    output_info["total_videos"] = total_videos
    output_info["total_tasks"] = 2
    output_info["total_chunks"] = max(1, math.ceil(new_episode_index / int(base_info["chunks_size"])))
    output_info["splits"] = {"train": f"0:{new_episode_index}"}
    _write_json_atomic(staging_destination / "meta/info.json", output_info)
    _write_jsonl_atomic(staging_destination / "meta/tasks.jsonl", TASKS)
    _write_jsonl_atomic(staging_destination / "meta/episodes.jsonl", output_episode_rows)

    report = {
        "schema_version": "openarm_kai0_awbc_v1",
        "destination": str(destination),
        "advantage_source": "relative_advantage",
        "stage_source": {
            "HQ_before_folding_only": "HQ-Stage absolute_value>=0.5",
            "HQ_folding_only": f"source_episode_index>={hq_folding_only_start} -> folding",
            "Site": "manual stage boundary" if site_annotations is not None else "HQ/Site-Stage absolute_value>=0.5",
            "TDA": "mapped from source HQ stage",
        },
        "hq_folding_only_start": hq_folding_only_start,
        "site_annotations": str(site_annotations_path.resolve()) if site_annotations_path is not None else None,
        "positive_ratio_target": positive_ratio,
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "relative_interval": relative_interval,
        "site_repeat": site_repeat,
        "unique_episode_counts": {
            "HQ": expected_hq_episodes,
            "Site": len(expected_site_ids),
            "TDA": len(tda_sources),
        },
        "materialized_episode_counts": {
            "HQ": expected_hq_episodes,
            "Site": len(expected_site_ids) * site_repeat,
            "TDA": len(tda_sources),
        },
        "total_episodes": new_episode_index,
        "total_frames": total_frames,
        "total_videos": total_videos,
        "unique_label_counts": unique_counts,
        "materialized_label_counts": materialized_counts,
        "overall_unique_label_counts": overall_unique,
        "site_validation_excluded": f"source_episode_index {site_train_source_end}:151",
        "site_failure_excluded": list(excluded_site_source_episodes),
        "source_ratio_bounds": list(source_ratio_bounds),
        "tasks": list(TASKS),
    }
    _write_json_atomic(staging_destination / "kai0_awbc_build_report.json", report)
    os.replace(staging_destination, destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hq-score-root", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--site-score-root", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--tda-augmented", required=True, type=pathlib.Path)
    parser.add_argument("--destination", required=True, type=pathlib.Path)
    parser.add_argument("--site-repeat", type=int, default=3)
    parser.add_argument("--tda-time-count", type=int, default=150)
    parser.add_argument("--tda-mirror-count", type=int, default=150)
    parser.add_argument("--positive-ratio", type=float, default=0.30)
    parser.add_argument("--relative-interval", type=int, default=50)
    parser.add_argument("--hq-folding-only-start", type=int, default=HQ_FOLDING_ONLY_START)
    parser.add_argument(
        "--site-annotations",
        type=pathlib.Path,
        required=True,
        help="Manual two-stage Site annotation JSONL; used only for stage-aware threshold groups.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_kai0_awbc_dataset(
        args.hq_score_root,
        args.site_score_root,
        args.tda_augmented,
        args.destination,
        site_repeat=args.site_repeat,
        tda_time_count=args.tda_time_count,
        tda_mirror_count=args.tda_mirror_count,
        positive_ratio=args.positive_ratio,
        relative_interval=args.relative_interval,
        overwrite=args.overwrite,
        hq_folding_only_start=args.hq_folding_only_start,
        site_annotations_path=args.site_annotations,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
