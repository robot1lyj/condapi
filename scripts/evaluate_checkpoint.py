"""Evaluate a YAM Pi0.5 checkpoint on held-out LeRobot episodes.

The report deliberately stays at the offline policy level: first-action MAE
and chunk continuity are useful diagnostics, but neither is a real-robot
success metric.  The dataset must expose the YAM 14D contract and all three
RGB streams.

Example::

    python scripts/evaluate_checkpoint.py \
        --config pi05_yam_lora \
        --checkpoint-dir ./checkpoints/pi05_yam_lora/lego_sorting/30000 \
        --dataset /path/to/yam_lerobot \
        --val-split 80:100 \
        --output ./eval_report
"""

import argparse
from collections import defaultdict
import dataclasses
import json
import logging
import pathlib
from typing import Any

import numpy as np

try:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config

logger = logging.getLogger(__name__)

YAM_ACTION_DIM = 14
YAM_IMAGE_KEYS = (
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
)


@dataclasses.dataclass
class EvalMetrics:
    episode_idx: int
    num_frames: int
    evaluated_frames: int
    mae_per_action: np.ndarray
    mae_overall: float
    temporal_consistency: list[float]
    temporal_consistency_mean: float


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _scalar_text(value: Any) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def _task_mapping(tasks: Any) -> dict[int, str]:
    """Normalize LeRobot v2/v3 task metadata to a simple lookup table."""
    if hasattr(tasks, "iterrows"):
        return {int(row["task_index"]): str(task) for task, row in tasks.iterrows()}
    return {int(index): str(task) for index, task in (tasks or {}).items()}


def _parse_episode_range(spec: str) -> list[int]:
    try:
        start, end = (int(value) for value in spec.split(":", maxsplit=1))
    except ValueError as error:
        raise ValueError(f"Expected episode range START:END, got {spec!r}.") from error
    if start < 0 or end <= start:
        raise ValueError(f"Episode range must satisfy 0 <= START < END, got {spec!r}.")
    return list(range(start, end))


def _first_action(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.ndim == 1:
        return action
    if action.ndim == 2:
        return action[0]
    raise ValueError(f"Expected an action vector or chunk, got shape={action.shape}.")


def _temporal_consistency(pred_chunks: list[np.ndarray]) -> list[float]:
    if len(pred_chunks) < 2:
        return []
    return [
        float(np.linalg.norm(pred_chunks[index][0] - pred_chunks[index + 1][0]))
        for index in range(len(pred_chunks) - 1)
    ]


def _build_observation(frame: dict, tasks: dict[int, str]) -> dict | None:
    """Build the raw observation expected by ``YamInputs``."""
    if "observation.state" not in frame or any(key not in frame for key in YAM_IMAGE_KEYS):
        return None

    observation = {
        "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
    }
    for key in YAM_IMAGE_KEYS:
        image = np.asarray(frame[key])
        if image.ndim != 3 or (image.shape[-1] != 3 and image.shape[0] != 3):
            return None
        observation[key] = image

    if "prompt" in frame:
        observation["prompt"] = _scalar_text(frame["prompt"])
    elif "task_index" in frame:
        task_index = _scalar_int(frame["task_index"])
        if task_index not in tasks:
            return None
        observation["prompt"] = tasks[task_index]
    elif "task" in frame:
        observation["prompt"] = _scalar_text(frame["task"])
    else:
        return None
    return observation


def evaluate_episode(
    policy: _policy.Policy,
    episode_idx: int,
    episode_frames: list[dict],
    tasks: dict[int, str],
) -> EvalMetrics:
    mae_accum = np.zeros(YAM_ACTION_DIM, dtype=np.float64)
    pred_chunks: list[np.ndarray] = []
    evaluated_frames = 0

    for frame_index, frame in enumerate(episode_frames):
        observation = _build_observation(frame, tasks)
        if observation is None:
            logger.warning("Skipping episode %s frame %s: incomplete YAM observation", episode_idx, frame_index)
            continue

        try:
            result = policy.infer(observation)
            predicted = np.asarray(result["actions"], dtype=np.float32)
            truth = _first_action(frame["action"])
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            logger.warning("Skipping episode %s frame %s: %s", episode_idx, frame_index, error)
            continue

        if predicted.ndim != 2 or predicted.shape[1] != YAM_ACTION_DIM or predicted.shape[0] == 0:
            raise ValueError(f"YAM policy returned invalid action shape {predicted.shape}; expected [horizon, 14].")
        if truth.shape != (YAM_ACTION_DIM,):
            raise ValueError(f"YAM dataset returned invalid action shape {truth.shape}; expected (14,).")
        if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
            raise ValueError("YAM action contains NaN or Inf.")

        pred_chunks.append(predicted)
        mae_accum += np.abs(predicted[0] - truth)
        evaluated_frames += 1

    if evaluated_frames == 0:
        raise ValueError(f"No valid YAM frames were evaluated in episode {episode_idx}.")

    mae_per_action = mae_accum / evaluated_frames
    temporal_consistency = _temporal_consistency(pred_chunks)
    return EvalMetrics(
        episode_idx=episode_idx,
        num_frames=len(episode_frames),
        evaluated_frames=evaluated_frames,
        mae_per_action=mae_per_action,
        mae_overall=float(mae_per_action.mean()),
        temporal_consistency=temporal_consistency,
        temporal_consistency_mean=float(np.mean(temporal_consistency)) if temporal_consistency else 0.0,
    )


def load_dataset(
    repo_id: str, episode_indices: list[int], action_horizon: int
) -> tuple[dict[int, list[dict]], dict[int, str]]:
    metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    action_dim = metadata.features["action"]["shape"][0]
    if action_dim != YAM_ACTION_DIM:
        raise ValueError(f"This evaluator is for YAM 14D data, but the dataset declares action_dim={action_dim}.")

    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [t / metadata.fps for t in range(action_horizon)]},
    )
    selected = set(episode_indices)
    episodes: dict[int, list[dict]] = defaultdict(list)
    for frame in dataset:
        episode_idx = _scalar_int(frame["episode_index"])
        if episode_idx in selected:
            episodes[episode_idx].append(dict(frame))
    return episodes, _task_mapping(metadata.tasks)


def _action_names() -> list[str]:
    return [
        *(f"left_j{i}" for i in range(1, 7)),
        "left_gripper",
        *(f"right_j{i}" for i in range(1, 7)),
        "right_gripper",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YAM Pi0.5 checkpoint on a held-out LeRobot split.")
    parser.add_argument("--config", default="pi05_yam_lora", help="YAM training config name")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to a complete checkpoint directory")
    parser.add_argument("--dataset", required=True, help="Path or repo id of a YAM LeRobot dataset")
    parser.add_argument("--val-split", required=True, help="Validation episode range, e.g. '80:100'")
    parser.add_argument("--output", default="./eval_report", help="Output directory for the JSON report")
    parser.add_argument("--max-episodes", type=int, default=0, help="Maximum episodes to evaluate (0=all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-episode metrics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_config = _config.get_config(args.config)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

    requested_episodes = _parse_episode_range(args.val_split)
    episodes, tasks = load_dataset(args.dataset, requested_episodes, train_config.model.action_horizon)
    episode_keys = sorted(episodes)
    if args.max_episodes > 0:
        episode_keys = episode_keys[: args.max_episodes]
    if not episode_keys:
        raise ValueError(f"No episodes from {args.val_split!r} were found in {args.dataset!r}.")

    metrics = [evaluate_episode(policy, episode_idx, episodes[episode_idx], tasks) for episode_idx in episode_keys]
    metrics.sort(key=lambda item: item.mae_overall, reverse=True)
    per_action = np.mean([item.mae_per_action for item in metrics], axis=0)
    action_names = _action_names()

    report = {
        "checkpoint": str(checkpoint_dir),
        "config": args.config,
        "dataset": args.dataset,
        "val_split": args.val_split,
        "num_val_episodes": len(metrics),
        "num_val_frames": sum(item.num_frames for item in metrics),
        "evaluated_frames": sum(item.evaluated_frames for item in metrics),
        "summary": {
            "mae_mean": float(np.mean([item.mae_overall for item in metrics])),
            "mae_median": float(np.median([item.mae_overall for item in metrics])),
            "mae_std": float(np.std([item.mae_overall for item in metrics])),
            "temporal_consistency_mean": float(np.mean([item.temporal_consistency_mean for item in metrics])),
        },
        "per_action_mae": dict(zip(action_names, (float(value) for value in per_action), strict=True)),
        "worst_10_episodes": [
            dataclasses.asdict(item) | {"mae_per_action": item.mae_per_action.tolist()} for item in metrics[:10]
        ],
    }
    report_path = pathlib.Path(args.output) / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    logger.info(
        "Evaluated %s episodes / %s frames; mean first-action MAE %.5f",
        len(metrics),
        report["evaluated_frames"],
        report["summary"]["mae_mean"],
    )
    if args.verbose:
        for item in metrics:
            print(
                f"episode={item.episode_idx} frames={item.evaluated_frames} "
                f"mae={item.mae_overall:.5f} temporal={item.temporal_consistency_mean:.5f}"
            )


if __name__ == "__main__":
    main()
