"""Backfill LeRobot v2.1 per-episode parquet stats.

Some LeRobot versions require ``meta/episodes_stats.jsonl`` even when OpenPI
does not directly use those stats. This script computes lightweight numeric
stats from parquet files only and does not decode videos.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import tqdm


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _parse_episodes(spec: str | None, info: dict[str, Any]) -> list[int]:
    if spec is None:
        return list(range(int(info["total_episodes"])))
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid episode range: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, end, step))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _episode_path(dataset_dir: pathlib.Path, info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunk = episode_index // int(info["chunks_size"])
    return dataset_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def _column_to_2d(series: pd.Series) -> np.ndarray | None:
    first_valid = next((value for value in series.to_numpy() if value is not None), None)
    if first_valid is None:
        return None

    if isinstance(first_valid, np.ndarray | list | tuple):
        array = np.stack([np.asarray(value) for value in series.to_numpy()])
        if not np.issubdtype(array.dtype, np.number):
            return None
        return array.reshape(len(series), -1)

    array = np.asarray(series.to_numpy())
    if not np.issubdtype(array.dtype, np.number):
        return None
    return array.reshape(len(series), -1)


def compute_episode_stats(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        values = _column_to_2d(frame[column])
        if values is None:
            continue
        values = values.astype(np.float64, copy=False)
        quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
        stats[column] = {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "count": [len(values)],
            "q01": quantiles[0],
            "q10": quantiles[1],
            "q50": quantiles[2],
            "q90": quantiles[3],
            "q99": quantiles[4],
        }
    return stats


def write_episode_stats(dataset_dir: pathlib.Path, *, episodes: str | None = None) -> pathlib.Path:
    dataset_dir = dataset_dir.resolve()
    info = _load_json(dataset_dir / "meta/info.json")
    rows = []
    for episode_index in tqdm.tqdm(_parse_episodes(episodes, info), desc="Computing episode stats"):
        parquet_path = _episode_path(dataset_dir, info, episode_index)
        frame = pd.read_parquet(parquet_path)
        rows.append({"episode_index": episode_index, "stats": compute_episode_stats(frame)})
    output = dataset_dir / "meta/episodes_stats.jsonl"
    _write_jsonl(output, rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", default=None, help="Optional range/list, e.g. 0:141 or 0,2,5.")
    args = parser.parse_args()

    output = write_episode_stats(args.dataset, episodes=args.episodes)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
