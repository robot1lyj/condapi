"""Split a LeRobot dataset into train/val and recompute norm stats with relative actions.

Usage:
    # Split 500 episodes for training, 200 for validation, and compute norm stats
    python scripts/split_and_norm_relative.py \
        --dataset /storage1t/datasets/high_quality_folding \
        --train-episodes 500 \
        --val-episodes 200 \
        --config pi05_openarms_dual

    # Only recompute norm stats (no split)
    python scripts/split_and_norm_relative.py \
        --dataset /storage1t/datasets/high_quality_folding \
        --no-split \
        --config pi05_openarms_dual

This script:
1. Optionally splits N episodes for train + val (sequentially from episode 0)
2. Recomputes norm stats using the RELATIVE action representation
   (UMI-style: each action is an offset from the current state)
3. Writes norm_stats.json to the dataset root
"""

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
import tqdm

# We import from openpi after setting up paths
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi import transforms as _transforms
from openpi.shared import normalize as _normalize
from openpi.shared import array_typing as at


def make_relative_actions(
    actions: np.ndarray,
    state: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Convert absolute actions to UMI-style relative actions.

    Each action in the chunk becomes an offset from the current state:
        action_rel[t+i] = action_abs[t+i] - state[t]

    This matches π0.5's pretraining distribution.
    """
    dims = mask.shape[-1]
    result = actions.copy()
    result[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
    return result


def make_bool_mask(*sizes: int) -> list[bool]:
    """Build a per-dim boolean mask. -1 means False (gripper)."""
    mask: list[bool] = []
    for s in sizes:
        if s == -1:
            mask.append(False)
        else:
            mask.extend([True] * s)
    return mask


def split_dataset(dataset_dir: pathlib.Path, train_n: int, val_n: int):
    """Update the dataset manifest to split train/val."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        print(f"Warning: {info_path} not found, cannot update splits")
        return

    info = json.loads(info_path.read_text())
    total_episodes = info.get("total_episodes", train_n + val_n)

    train_end = min(train_n, total_episodes)
    val_end = min(train_n + val_n, total_episodes)

    info["splits"] = {
        "train": f"0:{train_end}",
        "val": f"{train_end}:{val_end}",
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(f"Splits updated: train=0:{train_end}, val={train_end}:{val_end}")
    print(f"Total episodes: {total_episodes}, Train: {train_end}, Val: {val_end - train_end}")


def compute_relative_norm_stats(
    dataset_dir: pathlib.Path,
    config_name: str,
    max_frames: int | None = None,
):
    """Compute norm stats using relative actions."""
    import openpi.training.config as _config

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Load dataset
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(str(dataset_dir))
    dataset = lerobot_dataset.LeRobotDataset(
        str(dataset_dir),
        delta_timestamps={
            "action": [t / dataset_meta.fps for t in range(config.model.action_horizon)]
        },
    )

    action_dim = dataset_meta.features["action"]["shape"][0]
    mask = make_bool_mask(7, -1, 7, -1)  # 7 joints + 1 gripper per arm
    mask_np = np.asarray(mask)

    state_stats = _normalize.RunningStats()
    action_stats = _normalize.RunningStats()

    total_frames = len(dataset)
    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Computing relative-action norm stats over {total_frames} frames...")
    for i in tqdm.tqdm(range(0, total_frames, 128)):
        batch_end = min(i + 128, total_frames)
        states = []
        rel_actions = []

        for j in range(i, batch_end):
            frame = dataset[j]
            state = np.asarray(frame["observation.state"], dtype=np.float32)
            # action shape: [action_horizon, action_dim]
            action = np.asarray(frame["action"], dtype=np.float32)

            # Convert to UMI-relative
            state_1d = state.flatten()
            action_2d = action.reshape(1, *action.shape) if action.ndim == 2 else action
            rel_action = make_relative_actions(action_2d, state_1d, mask_np)

            states.append(state_1d)
            rel_actions.append(rel_action[0, 0])  # Use first action of chunk for stats

        if states:
            state_stats.update(np.stack(states))
            action_stats.update(np.stack(rel_actions))

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }

    # Write to dataset root (same location training expects)
    output_path = dataset_dir / "norm_stats.json"
    _normalize.save(dataset_dir, norm_stats)
    print(f"Relative-action norm stats written to {output_path}")

    return norm_stats


def main():
    parser = argparse.ArgumentParser(
        description="Split LeRobot dataset and/or recompute relative-action norm stats"
    )
    parser.add_argument("--dataset", required=True, help="Path to LeRobot dataset")
    parser.add_argument("--train-episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--val-episodes", type=int, default=200, help="Number of validation episodes")
    parser.add_argument("--no-split", action="store_true", help="Skip dataset splitting")
    parser.add_argument("--config", default="pi05_openarms_dual", help="Training config name")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames for norm stats")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset).resolve()
    if not dataset_dir.exists():
        print(f"Error: dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Step 1: Split dataset
    if not args.no_split:
        split_dataset(dataset_dir, args.train_episodes, args.val_episodes)
    else:
        print("Skipping dataset split (--no-split)")

    # Step 2: Compute relative-action norm stats
    compute_relative_norm_stats(dataset_dir, args.config, args.max_frames)


if __name__ == "__main__":
    main()
