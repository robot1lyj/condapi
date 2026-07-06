"""Rewrite LeRobot parquet timestamps from actual video FPS.

This fixes datasets whose tabular timestamps were generated from a nominal FPS
while MP4 files report a slightly different FPS. Video decoders can otherwise
map late-episode timestamps one frame past the end of the video.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np
import pandas as pd
import tqdm

from write_lerobot_episode_stats import write_episode_stats


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _parse_episodes(spec: str | None, total_episodes: int) -> list[int]:
    if spec is None:
        return list(range(total_episodes))
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid episode range: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, end, step))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _episode_path(dataset_dir: pathlib.Path, info: dict, episode_index: int) -> pathlib.Path:
    chunk = episode_index // int(info["chunks_size"])
    return dataset_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def _video_path(dataset_dir: pathlib.Path, info: dict, episode_index: int, video_key: str) -> pathlib.Path:
    chunk = episode_index // int(info["chunks_size"])
    return dataset_dir / info["video_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def _video_fps(path: pathlib.Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError(f"Could not read FPS from {path}")
    return fps


def rewrite_timestamps(
    dataset_dir: pathlib.Path,
    *,
    video_key: str,
    episodes: str | None,
    rewrite_episode_stats: bool,
) -> dict[str, float | int | str]:
    dataset_dir = dataset_dir.resolve()
    info = _load_json(dataset_dir / "meta/info.json")
    episode_indices = _parse_episodes(episodes, int(info["total_episodes"]))
    max_abs_delta = 0.0

    for episode_index in tqdm.tqdm(episode_indices, desc="Rewriting timestamps"):
        parquet_path = _episode_path(dataset_dir, info, episode_index)
        frame = pd.read_parquet(parquet_path)
        fps = _video_fps(_video_path(dataset_dir, info, episode_index, video_key))
        new_timestamps = (frame["frame_index"].to_numpy(dtype=np.float64) / fps).astype(np.float32)
        old_timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(old_timestamps - new_timestamps))))
        frame["timestamp"] = new_timestamps
        frame.to_parquet(parquet_path, index=False)

    if rewrite_episode_stats:
        write_episode_stats(dataset_dir)

    return {
        "dataset": str(dataset_dir),
        "video_key": video_key,
        "episodes": len(episode_indices),
        "max_abs_delta_seconds": max_abs_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--video-key", default="observation.images.base")
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--no-rewrite-episode-stats", action="store_true")
    args = parser.parse_args()

    report = rewrite_timestamps(
        args.dataset,
        video_key=args.video_key,
        episodes=args.episodes,
        rewrite_episode_stats=not args.no_rewrite_episode_stats,
    )
    (args.dataset / "timestamp_rewrite_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
