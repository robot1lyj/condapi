"""Offline generalization evaluation for VLA checkpoints.

Evaluates a trained pi05_openarms_dual checkpoint on a held-out validation
split of the dataset.  Produces a report with:

- Per-joint MAE vs ground truth (overall + per-episode)
- Temporal action consistency (L2 distance between consecutive predicted chunks)
- Cumulative rollout drift (autoregressive prediction over N steps)
- Per-episode ranking (worst-performing episodes first)

Usage:
    # Use train=0:N, val=N:end via the splits dict in dataset meta
    python scripts/evaluate_checkpoint.py \
        --config pi05_openarms_dual \
        --checkpoint-dir .../openarms_folding_v002_bs32_fsdp2/19999 \
        --dataset /share/home/linyongjia/datasets/openarms_folding_v002 \
        --val-split "132:165" \
        --output ./eval_report
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

# dataset loading
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvalMetrics:
    episode_idx: int
    num_frames: int

    # per-frame MAE averaged across joints
    mae_per_joint: np.ndarray  # [action_dim]
    mae_overall: float

    # temporal consistency: L2 between (pred[t] action chunk) and (pred[t+1] action chunk)
    # measured on the *first* action of each chunk to avoid comparing full 50-dim sequences
    temporal_consistency: list[float]  # one per adjacent frame pair
    temporal_consistency_mean: float

    # cumulative drift: autoregress over K consecutive frames, compare pred vs truth
    # drift[t][k] = L2(pred_{t+k}, truth_{t+k}) when using pred at t as state
    rollout_drift: list[float]  # drift at each autoregressive depth
    rollout_drift_mean: float


def _parse_split_spec(spec: str, total: int) -> tuple[list[int], list[int]]:
    """Parse a split spec like '0:132' or '132:165' into train and val indices."""
    train_spec, val_spec = spec.split(",")
    train_start, train_end = map(int, train_spec.split(":"))
    val_start, val_end = map(int, val_spec.split(":"))

    train_indices = list(range(train_start, min(train_end, total)))
    val_indices = list(range(val_start, min(val_end, total)))
    return train_indices, val_indices


def _action_mae(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-joint MAE. Both arrays shape [action_dim]."""
    return np.abs(pred - truth)


def _temporal_consistency(pred_chunks: list[np.ndarray]) -> list[float]:
    """Compute L2 between first action of consecutive predicted chunks.

    Args:
        pred_chunks: list of predicted action arrays, each shape [action_horizon, action_dim]

    Returns:
        list of L2 distances (len = len(pred_chunks) - 1)
    """
    if len(pred_chunks) < 2:
        return []
    diffs = []
    for i in range(len(pred_chunks) - 1):
        # Compare first action of chunk i vs first action of chunk i+1
        diff = np.linalg.norm(pred_chunks[i][0] - pred_chunks[i + 1][0])
        diffs.append(float(diff))
    return diffs


def evaluate_episode(
    policy: _policy.Policy,
    episode_frames: list[dict],
    action_dim: int,
    rollout_depth: int = 10,
    tasks: dict[int, str] | None = None,
) -> EvalMetrics:
    """Evaluate a single episode.

    Args:
        policy: loaded policy with transforms.
        episode_frames: list of raw dataset frames for one episode.
        action_dim: expected action dimension.
        rollout_depth: how many steps to autoregress for drift measurement.
        tasks: mapping from task_index to task description string.
    """
    mae_accum = np.zeros(action_dim)
    mae_count = 0
    pred_chunks = []
    rollout_errors = []

    for t, frame in enumerate(episode_frames):
        # Extract observation
        obs = _build_observation(frame, tasks=tasks)
        if obs is None:
            continue

        try:
            # Run inference
            result = policy.infer(obs)
        except Exception as exc:
            logger.warning(f"Episode inference error at frame {t}: {exc}")
            continue

        pred_actions = result.get("actions")
        if pred_actions is None:
            continue

        # pred_actions shape: [action_horizon, action_dim]
        pred_chunks.append(np.asarray(pred_actions))

        # Ground truth action (first frame of action chunk)
        if "action" in frame:
            truth_action = np.asarray(frame["action"][0], dtype=np.float32)
            if truth_action.shape[0] == action_dim:
                mae_accum += _action_mae(pred_actions[0], truth_action)
                mae_count += 1

        # Autoregressive rollout drift: use pred[0] as state, compare pred[t+
        ...

    # Compute aggregated metrics
    per_joint_mae = mae_accum / max(mae_count, 1)
    overall_mae = float(np.mean(per_joint_mae))

    tc = _temporal_consistency(pred_chunks)
    tc_mean = float(np.mean(tc)) if tc else 0.0

    rd_mean = float(np.mean(rollout_errors)) if rollout_errors else 0.0

    return EvalMetrics(
        episode_idx=0,
        num_frames=len(episode_frames),
        mae_per_joint=per_joint_mae,
        mae_overall=overall_mae,
        temporal_consistency=tc,
        temporal_consistency_mean=tc_mean,
        rollout_drift=rollout_errors,
        rollout_drift_mean=rd_mean,
    )


# Supported base camera key names — order = detection priority
_BASE_CAMERA_CANDIDATES = [
    "observation.images.top_rgb",
    "observation.images.base",
]


def _detect_image_keys(frame: dict) -> dict[str, str]:
    """Detect which camera keys are present in the dataset frame.

    Returns a mapping {src_key: dst_key} for all image keys found.
    The base camera is auto-detected from known candidates.
    """
    mappings: dict[str, str] = {}

    # Detect base camera
    for candidate in _BASE_CAMERA_CANDIDATES:
        if candidate in frame:
            mappings[candidate] = candidate
            break

    # Wrist cameras (same naming across all known datasets)
    for key in ("observation.images.left_wrist", "observation.images.right_wrist"):
        if key in frame:
            mappings[key] = key

    return mappings


def _build_observation(frame: dict, tasks: dict[int, str] | None = None) -> dict | None:
    """Build an observation dict from a raw dataset frame.

    Auto-detects camera keys (supports both top_rgb and base naming).
    Converts task_index to prompt string using dataset task metadata.
    """
    try:
        obs = {}
        # State
        if "observation.state" in frame:
            obs["observation.state"] = np.asarray(frame["observation.state"], dtype=np.float32)

        # Images — auto-detect which camera keys are present
        for src_key, dst_key in _detect_image_keys(frame).items():
            img = np.asarray(frame[src_key])
            if img.ndim == 3 and img.shape[-1] == 3:
                obs[dst_key] = img.astype(np.uint8)

        # Prompt: prefer explicit prompt field, then task_index + metadata, then task
        if "prompt" in frame:
            obs["prompt"] = str(frame["prompt"])
        elif "task_index" in frame and tasks is not None:
            task_idx = int(frame["task_index"])
            obs["prompt"] = tasks.get(task_idx, "fold the t-shirt")
        elif "task" in frame:
            obs["prompt"] = str(frame["task"])

        if "observation.state" not in obs:
            return None
        return obs
    except Exception:
        return None


def load_val_dataset(
    repo_id: str,
    val_indices: list[int],
    action_horizon: int = 50,
) -> tuple[list[dict], int, dict[int, str]]:
    """Load the validation portion of a LeRobot dataset.

    Returns:
        val_frames: list of frames belonging to val episodes.
        action_dim: dimension of action space.
        tasks: mapping from task_index to task description string.
    """
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    # Use delta_timestamps to get action chunks for ground-truth comparison
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            "action": [t / dataset_meta.fps for t in range(action_horizon)],
        },
    )

    action_dim = dataset_meta.features["action"]["shape"][0]

    # Group frames by episode
    val_frames = []
    for idx, frame in enumerate(dataset):
        episode_idx = int(frame.get("episode_index", 0))
        if episode_idx in val_indices:
            # Convert keys: lerobot stores as observation.state, action, etc.
            val_frames.append(dict(frame))

    return val_frames, action_dim, dataset_meta.tasks


def main():
    parser = argparse.ArgumentParser(description="VLA offline generalization evaluation")
    parser.add_argument("--config", required=True, help="Training config name, e.g. pi05_openarms_dual")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--dataset", required=True, help="Path to LeRobot dataset")
    parser.add_argument("--val-split", default="132:165", help="val episode range, e.g. '132:165'")
    parser.add_argument("--output", default="./eval_report", help="Output directory for report")
    parser.add_argument("--rollout-depth", type=int, default=10, help="Autoregressive rollout steps")
    parser.add_argument("--max-episodes", type=int, default=0, help="Max val episodes to evaluate (0=all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-episode details")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load config
    train_config = _config.get_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    # Create policy
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
    logger.info(f"Loaded policy from {checkpoint_dir}")

    # Load dataset and split
    val_indices = list(range(*map(int, args.val_split.split(":"))))
    logger.info(f"Loading dataset from {args.dataset}")
    val_frames, action_dim, tasks = load_val_dataset(
        args.dataset, val_indices, action_horizon=train_config.model.action_horizon
    )

    # Group by episode
    from collections import defaultdict

    episodes: dict[int, list[dict]] = defaultdict(list)
    for frame in val_frames:
        ep = int(frame.get("episode_index", -1))
        episodes[ep].append(frame)

    logger.info(f"Validation: {len(episodes)} episodes, {len(val_frames)} frames")
    if args.max_episodes > 0:
        episode_keys = sorted(episodes.keys())[: args.max_episodes]
        episodes = {k: episodes[k] for k in episode_keys}

    # Evaluate each episode
    all_metrics: list[EvalMetrics] = []
    for ep_idx in sorted(episodes.keys()):
        frames = episodes[ep_idx]
        metrics = evaluate_episode(policy, frames, action_dim, args.rollout_depth, tasks=tasks)
        metrics.episode_idx = ep_idx
        all_metrics.append(metrics)

        if args.verbose:
            print(f"Episode {ep_idx:4d}: frames={metrics.num_frames:4d}, "
                  f"MAE={metrics.mae_overall:.4f}, "
                  f"TC={metrics.temporal_consistency_mean:.4f}, "
                  f"Drift={metrics.rollout_drift_mean:.4f}")

    # Sort by MAE descending (worst first)
    all_metrics.sort(key=lambda m: m.mae_overall, reverse=True)

    # Generate report
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_mae = np.array([m.mae_overall for m in all_metrics])
    all_tc = np.array([m.temporal_consistency_mean for m in all_metrics])
    all_drift = np.array([m.rollout_drift_mean for m in all_metrics])

    # Aggregate per-joint MAE
    joint_names = [
        "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7", "right_grip",
        "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7", "left_grip",
    ]
    per_joint_maes = np.array([m.mae_per_joint for m in all_metrics])  # [N, 16]
    mean_joint_mae = per_joint_maes.mean(axis=0)

    report = {
        "checkpoint": str(checkpoint_dir),
        "config": args.config,
        "dataset": args.dataset,
        "val_split": args.val_split,
        "num_val_episodes": len(all_metrics),
        "num_val_frames": sum(m.num_frames for m in all_metrics),
        "summary": {
            "mae_mean": float(np.mean(all_mae)),
            "mae_median": float(np.median(all_mae)),
            "mae_std": float(np.std(all_mae)),
            "mae_min": float(np.min(all_mae)),
            "mae_max": float(np.max(all_mae)),
            "temporal_consistency_mean": float(np.mean(all_tc)),
            "temporal_consistency_std": float(np.std(all_tc)),
            "rollout_drift_mean": float(np.mean(all_drift)),
            "rollout_drift_std": float(np.std(all_drift)),
        },
        "per_joint_mae": {name: float(v) for name, v in zip(joint_names, mean_joint_mae)},
        "worst_10_episodes": [
            {
                "episode_idx": m.episode_idx,
                "num_frames": m.num_frames,
                "mae": m.mae_overall,
                "temporal_consistency": m.temporal_consistency_mean,
                "rollout_drift": m.rollout_drift_mean,
            }
            for m in all_metrics[:10]
        ],
        "best_10_episodes": [
            {
                "episode_idx": m.episode_idx,
                "num_frames": m.num_frames,
                "mae": m.mae_overall,
                "temporal_consistency": m.temporal_consistency_mean,
                "rollout_drift": m.rollout_drift_mean,
            }
            for m in all_metrics[-10:]
        ],
    }

    # Write report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  VLA Generalization Evaluation Report")
    print("=" * 60)
    print(f"  Checkpoint:        {checkpoint_dir}")
    print(f"  Val episodes:      {len(all_metrics)}")
    print(f"  Val frames:        {sum(m.num_frames for m in all_metrics)}")
    print(f"  MAE (mean/median): {report['summary']['mae_mean']:.4f} / {report['summary']['mae_median']:.4f}")
    print(f"  MAE (min/max):     {report['summary']['mae_min']:.4f} / {report['summary']['mae_max']:.4f}")
    print(f"  Temp Consistency:  {report['summary']['temporal_consistency_mean']:.4f}")
    print(f"  Rollout Drift:     {report['summary']['rollout_drift_mean']:.4f}")
    print("-" * 60)
    print("  Per-Joint MAE:")
    for name, mae in report["per_joint_mae"].items():
        bar = "█" * int(min(mae * 20, 40))
        arm = "R" if name.startswith("right") else "L"
        joint = name.split("_")[-1]
        print(f"    {arm} {joint:5s}: {mae:.4f} {bar}")
    print("-" * 60)
    print("  Worst 3 episodes:")
    for m in all_metrics[:3]:
        print(f"    ep {m.episode_idx:4d}: MAE={m.mae_overall:.4f}, TC={m.temporal_consistency_mean:.4f}, Drift={m.rollout_drift_mean:.4f}")
    print("  Best 3 episodes:")
    for m in all_metrics[-3:]:
        print(f"    ep {m.episode_idx:4d}: MAE={m.mae_overall:.4f}, TC={m.temporal_consistency_mean:.4f}, Drift={m.rollout_drift_mean:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
