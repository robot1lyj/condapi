"""Repair a LeRobot v2.1 subset extracted from a larger dataset.

This script fixes the two most common subset issues:

1. The parquet filename/metadata episode index does not match the internal
   `episode_index` column stored inside the parquet.
2. The parquet contains trailing rows whose timestamps would decode to a video
   frame index outside the available MP4 frame range.

It rewrites each parquet in place, updates `meta/episodes.jsonl`, refreshes
`meta/info.json`, and writes a `meta/repair_report.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="LeRobot v2.1 dataset root directory.")
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="Video key to use as the authoritative frame-count source. Defaults to top_rgb if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print the planned changes without rewriting files.",
    )
    return parser.parse_args()


def load_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    return json.loads(info_path.read_text())


def discover_video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, value in info["features"].items() if isinstance(value, dict) and value.get("dtype") == "video"]


def choose_reference_video_key(video_keys: list[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in video_keys:
            raise ValueError(f"Video key {requested!r} not found. Available: {video_keys}")
        return requested
    if "observation.images.top_rgb" in video_keys:
        return "observation.images.top_rgb"
    if not video_keys:
        raise ValueError("No video keys found in dataset metadata.")
    return video_keys[0]


def count_video_frames(video_path: Path) -> int:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.frames and stream.frames > 0:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def compute_valid_length(timestamps: np.ndarray, fps: int, frame_count: int) -> int:
    frame_indices = np.rint(timestamps * fps).astype(int)
    invalid = np.nonzero(frame_indices >= frame_count)[0]
    if len(invalid) == 0:
        return len(timestamps)
    return int(invalid[0])


def parquet_files(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))


def parse_episode_id(parquet_path: Path) -> int:
    return int(parquet_path.stem.split("_")[1])


def parse_chunk_id(parquet_path: Path) -> int:
    return int(parquet_path.parent.name.split("-")[1])


def resolve_video_path(dataset_dir: Path, chunk_id: int, episode_id: int, video_key: str) -> Path:
    return dataset_dir / "videos" / f"chunk-{chunk_id:03d}" / video_key / f"episode_{episode_id:06d}.mp4"


def rewrite_parquet(parquet_path: Path, episode_id: int, valid_len: int, *, dry_run: bool) -> tuple[int, int]:
    df = pl.read_parquet(parquet_path)
    original_len = df.height
    cols = []
    for col in df.columns:
        if col == "episode_index":
            cols.append(pl.lit(episode_id, dtype=df.schema[col]).alias(col))
        else:
            cols.append(pl.col(col))
    df = df.select(cols).head(valid_len)
    if not dry_run:
        df.write_parquet(parquet_path)
    return original_len, df.height


def load_episode_lines(episodes_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]


def save_episode_lines(episodes_path: Path, lines: list[dict[str, Any]], *, dry_run: bool) -> None:
    if dry_run:
        return
    episodes_path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n")


def save_info(info_path: Path, info: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


def save_report(report_path: Path, report: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    info = load_info(dataset_dir)
    fps = int(info["fps"])
    video_keys = discover_video_keys(info)
    ref_video_key = choose_reference_video_key(video_keys, args.video_key)

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    episode_lines = load_episode_lines(episodes_path)
    by_episode = {line["episode_index"]: line for line in episode_lines}

    changes: list[dict[str, Any]] = []
    total_frames = 0
    files = parquet_files(dataset_dir)

    for parquet_path in files:
        episode_id = parse_episode_id(parquet_path)
        chunk_id = parse_chunk_id(parquet_path)
        video_path = resolve_video_path(dataset_dir, chunk_id, episode_id, ref_video_key)
        if not video_path.exists():
            raise FileNotFoundError(f"Missing reference video: {video_path}")

        timestamps = pl.read_parquet(parquet_path, columns=["timestamp"])["timestamp"].to_numpy()
        frame_count = count_video_frames(video_path)
        valid_len = compute_valid_length(timestamps, fps, frame_count)
        original_len, new_len = rewrite_parquet(parquet_path, episode_id, valid_len, dry_run=args.dry_run)
        total_frames += new_len

        if episode_id not in by_episode:
            raise KeyError(f"Episode {episode_id} missing from meta/episodes.jsonl")
        by_episode[episode_id]["length"] = new_len

        changes.append(
            {
                "episode_id": episode_id,
                "chunk_id": chunk_id,
                "reference_video_key": ref_video_key,
                "video_frames": frame_count,
                "original_rows": original_len,
                "repaired_rows": new_len,
                "trimmed_rows": original_len - new_len,
            }
        )

    repaired_lines = [by_episode[idx] for idx in sorted(by_episode)]
    save_episode_lines(episodes_path, repaired_lines, dry_run=args.dry_run)

    info["total_episodes"] = len(files)
    info["total_frames"] = total_frames
    info["total_videos"] = len(files) * len(video_keys)
    info["splits"] = {"train": f"0:{len(files)}"}
    save_info(dataset_dir / "meta" / "info.json", info, dry_run=args.dry_run)

    report = {
        "dataset_dir": str(dataset_dir),
        "fps": fps,
        "reference_video_key": ref_video_key,
        "video_keys": video_keys,
        "total_episodes": len(files),
        "total_frames": total_frames,
        "changes": changes,
    }
    save_report(dataset_dir / "meta" / "repair_report.json", report, dry_run=args.dry_run)

    trimmed = sum(change["trimmed_rows"] for change in changes)
    print(f"Repaired {len(files)} episodes in {dataset_dir}")
    print(f"Reference video key: {ref_video_key}")
    print(f"Total trimmed rows: {trimmed}")
    if args.dry_run:
        print("Dry run only; no files were rewritten.")


if __name__ == "__main__":
    main()
