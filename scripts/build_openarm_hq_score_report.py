"""Build an interactive KAI0-style report from completed HQ-Score shard outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd
import tqdm

try:
    from scripts.openarm_advantage_report import StageReportConfig
    from scripts.openarm_advantage_report import write_stage_report
    from scripts.openarm_kai0_contract import HQ_FOLDING_ONLY_START
except ModuleNotFoundError:
    from openarm_advantage_report import StageReportConfig
    from openarm_advantage_report import write_stage_report
    from openarm_kai0_contract import HQ_FOLDING_ONLY_START


VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
SHARDS = (
    ("s0_000_167", 0, 167),
    ("s1_167_334", 167, 334),
    ("s2_334_501", 334, 501),
    ("s3_501_668", 501, 668),
    ("s4_668_835", 668, 835),
    ("s5_835_999", 835, 999),
)
EPISODE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_video_path(info: dict[str, Any], episode_index: int, video_key: str) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(
        info["video_path"].format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)
    )


def _link_video(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _local_episode_index(path: pathlib.Path) -> int:
    match = EPISODE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Invalid episode parquet name: {path}")
    return int(match.group(1))


def _stage_aware_progress(raw_progress: np.ndarray, source_episode: int) -> tuple[np.ndarray, np.ndarray, int | None]:
    if source_episode >= HQ_FOLDING_ONLY_START:
        progress = np.clip(0.5 + raw_progress, 0.5, 1.0).astype(np.float32)
        stage_ids = np.ones(len(progress), dtype=np.int64)
        return progress, stage_ids, 0
    progress = np.clip(raw_progress, 0.0, 1.0).astype(np.float32)
    stage_ids = (progress >= 0.5).astype(np.int64)
    crossings = np.flatnonzero(stage_ids == 1)
    boundary = int(crossings[0]) if crossings.size else None
    return progress, stage_ids, boundary


def build_hq_score_report(
    datasets_root: pathlib.Path,
    output_root: pathlib.Path,
    *,
    overwrite: bool,
) -> pathlib.Path:
    datasets_root = datasets_root.resolve()
    output_root = output_root.resolve()
    source = datasets_root / "high_quality_folding"
    source_info = _load_json(source / "meta/info.json")
    source_rows = {int(row["episode_index"]): row for row in _load_jsonl(source / "meta/episodes.jsonl")}
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    report_dir = output_root / "hq_score_report"
    report_dir.mkdir(parents=True)

    payloads = []
    shard_counts = {}
    for shard_name, source_start, source_end in SHARDS:
        shard_root = datasets_root / f"openarm_kai0_stage_scores_hq_v1_{shard_name}"
        parquet_files = sorted((shard_root / "data").rglob("episode_*.parquet"))
        shard_counts[shard_name] = len(parquet_files)
        for parquet_path in tqdm.tqdm(parquet_files, desc=shard_name, leave=False):
            local_episode = _local_episode_index(parquet_path)
            source_episode = source_start + local_episode
            if source_episode >= source_end:
                raise ValueError(f"Shard {shard_name} produced out-of-range episode {local_episode}")
            frame = pd.read_parquet(
                parquet_path,
                columns=["relative_advantage", "absolute_value", "absolute_advantage"],
            )
            raw_progress = frame["absolute_value"].to_numpy(dtype=np.float32)
            relative_advantage = frame["relative_advantage"].to_numpy(dtype=np.float32)
            progress, stage_ids, boundary = _stage_aware_progress(raw_progress, source_episode)
            video_paths = {}
            video_aspect_ratios = {}
            for video_key in VIDEO_KEYS:
                source_video = source / _format_video_path(source_info, source_episode, video_key)
                camera_name = video_key.removeprefix("observation.images.")
                destination_video = output_root / "videos" / camera_name / f"episode_{source_episode:06d}.mp4"
                _link_video(source_video, destination_video)
                video_paths[camera_name] = f"../videos/{camera_name}/episode_{source_episode:06d}.mp4"
                shape = source_info["features"][video_key].get("shape", [])
                if len(shape) >= 2 and int(shape[0]) > 0:
                    video_aspect_ratios[camera_name] = float(shape[1]) / float(shape[0])

            source_row = source_rows[source_episode]
            payload = {
                "episode_index": source_episode,
                "length": len(frame),
                "fps": int(source_info.get("fps", 30)),
                "duration_s": (len(frame) - 1) / float(source_info.get("fps", 30)),
                "quality": "predicted",
                "eligible_for_k_data": True,
                "flatten_done_frame": boundary,
                "source_dataset": "HQ",
                "source_episode_index": source_episode,
                "source_tasks": source_row.get("tasks", []),
                "progress": np.round(progress, 6).tolist(),
                "episode_relative_progress": np.round(raw_progress, 6).tolist(),
                "stage_id": stage_ids.tolist(),
                "advantage": np.round(relative_advantage, 6).tolist(),
                "advantage_min": float(relative_advantage.min()),
                "advantage_mean": float(relative_advantage.mean()),
                "advantage_max": float(relative_advantage.max()),
                "negative_fraction": float(np.mean(relative_advantage < 0)),
                "videos": video_paths,
                "video_aspect_ratios": video_aspect_ratios,
                "score_source": "HQ-Stage",
                "score_shard": shard_name,
            }
            payloads.append(payload)

    payloads.sort(key=lambda payload: int(payload["episode_index"]))
    if not payloads:
        raise ValueError("No completed HQ-Score episode parquets found")
    summary = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "score_source": "HQ-Stage",
        "advantage_source": "relative_advantage",
        "progress_source": "absolute_value with a 0.5 folding-only stage offset for HQ 536:999",
        "hq_folding_only_start": HQ_FOLDING_ONLY_START,
        "completed_episodes": len(payloads),
        "completed_frames": sum(int(payload["length"]) for payload in payloads),
        "shard_counts": shard_counts,
        "negative_frame_fraction": float(
            np.average(
                [payload["negative_fraction"] for payload in payloads],
                weights=[payload["length"] for payload in payloads],
            )
        ),
    }
    index_path = write_stage_report(
        report_dir,
        payloads,
        summary,
        StageReportConfig(
            title="HQ-Stage Trajectory Review",
            subtitle="Stage-aware task progress and direct paired-frame advantage from completed HQ episodes.",
            score_source="HQ-Stage / checkpoint 10000",
            progress_title="Task progress / folding-only episodes start at 0.5",
            advantage_title="Direct advantage / Stage(frame t, frame t+50)",
        ),
    )
    (output_root / "hq_score_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(build_hq_score_report(args.datasets_root, args.output_root, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
