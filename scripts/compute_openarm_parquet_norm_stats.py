"""Compute OpenArm norm stats from LeRobot v2.1 parquet files only.

This avoids decoding videos during normalization. It matches the OpenArm
`pi05_openarms_dual_hq_tda_aug` data transform: state is absolute 16D, actions
are UMI-style relative chunks where joint dimensions are offset from the current
state and gripper dimensions are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import tqdm

from openpi.shared import normalize as _normalize
from openpi.transforms import make_bool_mask


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


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
) -> dict[str, _normalize.NormStats]:
    state_stats = _normalize.RunningStats()
    action_stats = _normalize.RunningStats()
    mask = np.asarray(make_bool_mask(7, -1, 7, -1))

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
    return {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }


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
    _normalize.save(output_dir, norm_stats)
    print(f"Wrote {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
