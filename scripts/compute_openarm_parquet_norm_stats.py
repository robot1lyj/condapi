"""Compute OpenArm norm stats from LeRobot v2.1 parquet files only.

This avoids decoding videos during normalization. It matches the OpenArm
`pi05_openarms_dual_hq_tda_aug` data transform: state is absolute 16D, actions
are UMI-style relative chunks where joint dimensions are offset from the current
state and gripper dimensions are left unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import pathlib

import numpy as np
import pandas as pd
import tqdm


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json_atomic(path: pathlib.Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _episode_path(dataset_dir: pathlib.Path, info: dict, episode_index: int) -> pathlib.Path:
    chunk = episode_index // int(info["chunks_size"])
    return dataset_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def _parse_episodes(spec: str | None, info: dict) -> list[int]:
    if spec is None:
        split = info.get("splits", {}).get("train")
        if split is None:
            return list(range(int(info["total_episodes"])))
        spec = split

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid episode range: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, end, step))

    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _stack_column(values: pd.Series, *, expected_dim: int, key: str) -> np.ndarray:
    array = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values.to_numpy()])
    if array.shape[-1] != expected_dim:
        raise ValueError(f"{key} expected {expected_dim}D, got {array.shape[-1]}D")
    return array


def _make_bool_mask(*dims: int) -> np.ndarray:
    mask: list[bool] = []
    for dim in dims:
        mask.extend([dim > 0] * abs(dim))
    return np.asarray(mask, dtype=bool)


@dataclass
class _Stats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray

    def to_json(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
        }


class _RunningStats:
    """OpenPI-compatible running stats without importing the full OpenPI stack."""

    def __init__(self, *, bins: int = 5000):
        self._bins = bins
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float32).reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if num_elements == 0:
            return

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._bins, dtype=np.float64) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._bins + 1)
                for i in range(vector_length)
            ]
        else:
            assert self._mean is not None
            assert self._mean_of_squares is not None
            assert self._min is not None
            assert self._max is not None
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements
        assert self._mean is not None
        assert self._mean_of_squares is not None
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)
        self._update_histograms(batch)

    def get_statistics(self) -> _Stats:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        assert self._mean is not None
        assert self._mean_of_squares is not None
        variance = self._mean_of_squares - self._mean**2
        std = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return _Stats(mean=self._mean, std=std, q01=q01, q99=q99)

    def _adjust_histograms(self) -> None:
        assert self._histograms is not None
        assert self._bin_edges is not None
        assert self._min is not None
        assert self._max is not None
        for i, hist in enumerate(self._histograms):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._bins + 1)
            self._histograms[i], _ = np.histogram(old_edges[:-1], bins=new_edges, weights=hist)
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        assert self._histograms is not None
        assert self._bin_edges is not None
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        assert self._histograms is not None
        assert self._bin_edges is not None
        results = []
        for quantile in quantiles:
            target_count = quantile * self._count
            values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumulative = np.cumsum(hist)
                idx = min(np.searchsorted(cumulative, target_count), len(edges) - 1)
                values.append(edges[idx])
            results.append(np.asarray(values))
        return results


def _relative_action_chunks(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    action_horizon: int,
    mask: np.ndarray,
) -> np.ndarray:
    frame_count = len(actions)
    indices = np.arange(frame_count)[:, None] + np.arange(action_horizon)[None, :]
    indices = np.minimum(indices, frame_count - 1)
    chunks = actions[indices].copy()
    chunks[..., mask] -= states[:, None, mask]
    return chunks


def compute_norm_stats(
    dataset_dir: pathlib.Path,
    *,
    action_horizon: int,
    action_dim: int,
    episodes: list[int],
    max_frames: int | None,
) -> dict[str, _Stats]:
    state_stats = _RunningStats()
    action_stats = _RunningStats()
    mask = _make_bool_mask(7, -1, 7, -1)

    total_frames = 0
    info = _load_json(dataset_dir / "meta/info.json")
    for episode_index in tqdm.tqdm(episodes, desc="Computing parquet norm stats"):
        parquet_path = _episode_path(dataset_dir, info, episode_index)
        df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
        if max_frames is not None:
            remaining = max_frames - total_frames
            if remaining <= 0:
                break
            df = df.iloc[:remaining]

        states = _stack_column(df["observation.state"], expected_dim=action_dim, key="observation.state")
        actions = _stack_column(df["action"], expected_dim=action_dim, key="action")
        relative_actions = _relative_action_chunks(
            actions,
            states,
            action_horizon=action_horizon,
            mask=mask,
        )

        state_stats.update(states)
        action_stats.update(relative_actions)
        total_frames += len(df)

    if total_frames == 0:
        raise ValueError("No frames processed")

    print(f"Processed {total_frames} frames from {len(episodes)} candidate episodes")
    return {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", default=None, help="Episode range/list. Defaults to meta/info train split.")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    info = _load_json(dataset_dir / "meta/info.json")
    episodes = _parse_episodes(args.episodes, info)
    meta_episodes = _load_jsonl(dataset_dir / "meta/episodes.jsonl")
    if len(meta_episodes) < max(episodes) + 1:
        raise ValueError("Episode spec exceeds meta/episodes.jsonl")

    norm_stats = compute_norm_stats(
        dataset_dir,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        episodes=episodes,
        max_frames=args.max_frames,
    )

    output_dir = args.output_dir.resolve() if args.output_dir is not None else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_path = output_dir / "norm_stats.json"
    _write_json_atomic(
        norm_path,
        {"norm_stats": {key: stats.to_json() for key, stats in norm_stats.items()}},
    )
    print(f"Wrote {norm_path}")


if __name__ == "__main__":
    main()
