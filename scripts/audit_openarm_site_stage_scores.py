"""Audit HQ-Stage transfer on labeled Site episodes and build a video-aligned report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.openarm_advantage_report import StageReportConfig
    from scripts.openarm_advantage_report import write_stage_report
    from scripts.openarm_stage_progress import build_stage_arrays
    from scripts.openarm_stage_progress import normalize_boundaries
except ModuleNotFoundError:
    from openarm_advantage_report import StageReportConfig
    from openarm_advantage_report import write_stage_report
    from openarm_stage_progress import build_stage_arrays
    from openarm_stage_progress import normalize_boundaries


VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


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


def _link_video(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        return
    except OSError:
        shutil.copy2(source, destination)


def _safe_corrcoef(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2 or float(prediction.std()) < 1e-12 or float(target.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def _reference_relative(progress: np.ndarray, interval: int) -> np.ndarray:
    result = np.zeros_like(progress, dtype=np.float32)
    for frame_index in range(len(progress)):
        future_index = min(frame_index + interval, len(progress) - 1)
        if future_index == frame_index:
            continue
        delta = future_index - frame_index
        scale = interval / delta if delta != interval else 1.0
        result[frame_index] = (progress[future_index] - progress[frame_index]) * scale
    return result


def audit_site_scores(
    source: pathlib.Path,
    annotations_path: pathlib.Path,
    score_roots: list[pathlib.Path],
    output_root: pathlib.Path,
    *,
    expected_episodes: int,
    relative_interval: int,
    overwrite: bool,
) -> dict[str, Any]:
    source = source.resolve()
    output_root = output_root.resolve()
    source_info = _load_json(source / "meta/info.json")
    annotations = {int(row["episode_index"]): row for row in _load_jsonl(annotations_path)}
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    payloads: list[dict[str, Any]] = []
    absolute_predictions = []
    absolute_targets = []
    relative_predictions = []
    relative_targets = []
    boundary_errors = []
    per_episode = []
    seen_sources: set[int] = set()

    for score_root in sorted(path.resolve() for path in score_roots):
        score_info = _load_json(score_root / "meta/info.json")
        for row in _load_jsonl(score_root / "meta/episodes.jsonl"):
            local_episode = int(row["episode_index"])
            source_episode = int(row["source_episode_index"])
            if source_episode in seen_sources:
                raise ValueError(f"Duplicate source episode across score shards: {source_episode}")
            seen_sources.add(source_episode)
            annotation = annotations.get(source_episode)
            if annotation is None:
                raise ValueError(f"Missing annotation for scored Site episode {source_episode}")
            if str(annotation.get("quality", "success")) != "success":
                raise ValueError(f"Non-success Site episode was scored: {source_episode}")

            frame = pd.read_parquet(
                score_root / _format_data_path(score_info, local_episode),
                columns=["relative_advantage", "absolute_value", "absolute_advantage"],
            )
            length = len(frame)
            boundaries = normalize_boundaries(annotation, length)
            reference_progress, _ = build_stage_arrays(length, boundaries)
            reference_relative = _reference_relative(reference_progress, relative_interval)
            predicted_progress = frame["absolute_value"].to_numpy(dtype=np.float32)
            predicted_relative = frame["relative_advantage"].to_numpy(dtype=np.float32)
            if not np.isfinite(predicted_progress).all() or not np.isfinite(predicted_relative).all():
                raise ValueError(f"Site score episode {source_episode} contains non-finite values")

            crossings = np.flatnonzero(predicted_progress >= 0.5)
            predicted_boundary = int(crossings[0]) if crossings.size else None
            reference_boundary = int(boundaries[0]["end_frame"])
            boundary_error = (
                abs(predicted_boundary - reference_boundary) / max(1, length - 1)
                if predicted_boundary is not None
                else 1.0
            )
            boundary_errors.append(boundary_error)

            absolute_error = predicted_progress - reference_progress
            relative_error = predicted_relative - reference_relative
            direction_mask = np.abs(reference_relative) >= 0.005
            direction_accuracy = (
                float(np.mean((predicted_relative[direction_mask] >= 0) == (reference_relative[direction_mask] >= 0)))
                if np.any(direction_mask)
                else 0.0
            )
            per_episode.append(
                {
                    "source_episode_index": source_episode,
                    "length": length,
                    "absolute_mse": float(np.mean(np.square(absolute_error))),
                    "absolute_mae": float(np.mean(np.abs(absolute_error))),
                    "relative_mse": float(np.mean(np.square(relative_error))),
                    "relative_mae": float(np.mean(np.abs(relative_error))),
                    "direction_accuracy": direction_accuracy,
                    "reference_boundary_frame": reference_boundary,
                    "predicted_boundary_frame": predicted_boundary,
                    "boundary_error_fraction": boundary_error,
                }
            )

            video_paths = {}
            video_aspect_ratios = {}
            for video_key in VIDEO_KEYS:
                camera_name = video_key.removeprefix("observation.images.")
                source_video = source / _format_video_path(source_info, source_episode, video_key)
                destination_video = output_root / "videos" / camera_name / f"episode_{source_episode:06d}.mp4"
                _link_video(source_video, destination_video)
                video_paths[camera_name] = f"../videos/{camera_name}/episode_{source_episode:06d}.mp4"
                shape = source_info["features"][video_key].get("shape", [])
                if len(shape) >= 2 and int(shape[0]) > 0:
                    video_aspect_ratios[camera_name] = float(shape[1]) / float(shape[0])

            stage_ids = (predicted_progress >= 0.5).astype(np.int64)
            payloads.append(
                {
                    "episode_index": source_episode,
                    "length": length,
                    "fps": int(source_info.get("fps", 30)),
                    "duration_s": (length - 1) / float(source_info.get("fps", 30)),
                    "quality": "success / human reference",
                    "eligible_for_k_data": source_episode < 141,
                    "flatten_done_frame": reference_boundary,
                    "predicted_flatten_done_frame": predicted_boundary,
                    "source_dataset": "Site-A150",
                    "source_episode_index": source_episode,
                    "progress": np.round(predicted_progress, 6).tolist(),
                    "reference_progress": np.round(reference_progress, 6).tolist(),
                    "stage_id": stage_ids.tolist(),
                    "advantage": np.round(predicted_relative, 6).tolist(),
                    "advantage_min": float(predicted_relative.min()),
                    "advantage_mean": float(predicted_relative.mean()),
                    "advantage_max": float(predicted_relative.max()),
                    "negative_fraction": float(np.mean(predicted_relative < 0)),
                    "videos": video_paths,
                    "video_aspect_ratios": video_aspect_ratios,
                    "score_source": "HQ-Stage direct transfer",
                }
            )

            absolute_predictions.append(predicted_progress)
            absolute_targets.append(reference_progress)
            relative_predictions.append(predicted_relative)
            relative_targets.append(reference_relative)

    if len(seen_sources) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} unique Site episodes, found {len(seen_sources)}")

    absolute_prediction = np.concatenate(absolute_predictions)
    absolute_target = np.concatenate(absolute_targets)
    relative_prediction = np.concatenate(relative_predictions)
    relative_target = np.concatenate(relative_targets)
    absolute_error = absolute_prediction - absolute_target
    relative_error = relative_prediction - relative_target
    sign_mask = np.abs(relative_target) >= 0.005
    ss_res = float(np.sum(np.square(absolute_error)))
    ss_tot = float(np.sum(np.square(absolute_target - absolute_target.mean())))
    metrics = {
        "absolute_mse": float(np.mean(np.square(absolute_error))),
        "absolute_mae": float(np.mean(np.abs(absolute_error))),
        "relative_mse": float(np.mean(np.square(relative_error))),
        "relative_mae": float(np.mean(np.abs(relative_error))),
        "direction_accuracy": float(
            np.mean((relative_prediction[sign_mask] >= 0) == (relative_target[sign_mask] >= 0))
        ),
        "corrcoef": _safe_corrcoef(absolute_prediction, absolute_target),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0,
        "boundary_error_median": float(np.median(boundary_errors)),
        "boundary_error_p90": float(np.percentile(boundary_errors, 90)),
        "episode_count": len(seen_sources),
        "frame_count": int(absolute_prediction.size),
    }
    gates = {
        "absolute_mse<=0.020": metrics["absolute_mse"] <= 0.020,
        "absolute_mae<=0.110": metrics["absolute_mae"] <= 0.110,
        "direction_accuracy>=0.88": metrics["direction_accuracy"] >= 0.88,
        "corrcoef>=0.92": metrics["corrcoef"] >= 0.92,
        "r2>=0.80": metrics["r2"] >= 0.80,
        "boundary_error_median<=0.05": metrics["boundary_error_median"] <= 0.05,
        "boundary_error_p90<=0.12": metrics["boundary_error_p90"] <= 0.12,
    }
    passed = all(gates.values())
    summary = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "score_source": "HQ-Stage direct transfer",
        "completed_episodes": len(seen_sources),
        "completed_frames": int(absolute_prediction.size),
        "negative_frame_fraction": float(np.mean(relative_prediction < 0)),
        "quality_gate_passed": passed,
        "metrics": metrics,
        "gates": gates,
    }
    report_dir = output_root / "site_score_report"
    report_index = write_stage_report(
        report_dir,
        payloads,
        summary,
        StageReportConfig(
            title="Site Stage Transfer Review",
            subtitle="HQ-Stage predictions over Site-A150 with the human two-stage reference shown in amber.",
            score_source="HQ-Stage direct transfer",
            progress_title="Predicted progress / amber is human stage reference",
            advantage_title="Direct advantage / Stage(frame t, frame t+50)",
        ),
    )
    result = {
        "source": str(source),
        "annotations": str(annotations_path),
        "score_roots": [str(path) for path in score_roots],
        "report_index": str(report_index),
        "passed": passed,
        "metrics": metrics,
        "gates": gates,
        "per_episode": sorted(per_episode, key=lambda row: int(row["source_episode_index"])),
    }
    _write_json_atomic(output_root / "site_stage_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--annotations", required=True, type=pathlib.Path)
    parser.add_argument("--score-root", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--expected-episodes", type=int, default=150)
    parser.add_argument("--relative-interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = audit_site_scores(
        args.source,
        args.annotations,
        args.score_root,
        args.output_root,
        expected_episodes=args.expected_episodes,
        relative_interval=args.relative_interval,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: result[key] for key in ("passed", "metrics", "gates", "report_index")}, indent=2))


if __name__ == "__main__":
    main()
