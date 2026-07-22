"""Sampled offline checkpoint sweep for OpenArm pi0.5 checkpoints.

The goal is not to replace real robot rollouts. This script produces cheap,
repeatable proxy metrics on held-out LeRobot episodes:

- action MAE/RMSE over predicted chunks
- first-step MAE
- gripper-only and joint-only MAE
- critical-frame MAE from high action/gripper-change frames
- chunk overlap consistency for adjacent sampled frames

It also evaluates a deterministic train subset so train/val gaps can flag
possible memorization.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import csv
import dataclasses
import gc
import json
import logging
import os
import pathlib
import re
import time
from typing import Any

import cv2
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pandas as pd

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config

LOGGER = logging.getLogger("openarm_checkpoint_sweep")
REPORT_SCHEMA_VERSION = "openarm_checkpoint_sweep_v2"
CRITICAL_SELECTOR_VERSION = "joint_action_and_gripper_frame_delta_v1"
VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


@dataclasses.dataclass(frozen=True)
class SampleSpec:
    episode_index: int
    frame_offset: int
    dataset_index: int
    kind: str


def _parse_range(spec: str) -> list[int]:
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid range: {spec}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, end, step))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _subsample_episodes(episodes: list[int], max_episodes: int, *, seed: int) -> list[int]:
    if max_episodes <= 0 or max_episodes >= len(episodes):
        return episodes
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(np.asarray(episodes), size=max_episodes, replace=False))
    return [int(x) for x in selected]


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json_atomic(path: pathlib.Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _episode_parquet_path(dataset_dir: pathlib.Path, episode_index: int) -> pathlib.Path:
    info = _load_json(dataset_dir / "meta/info.json")
    chunk = episode_index // int(info["chunks_size"])
    return dataset_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def _video_path(dataset_dir: pathlib.Path, episode_index: int, video_key: str) -> pathlib.Path:
    info = _load_json(dataset_dir / "meta/info.json")
    chunk = episode_index // int(info["chunks_size"])
    return dataset_dir / info["video_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def _video_frame_count(path: pathlib.Path) -> int:
    for attempt in range(3):
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if count > 0:
            return count
        if attempt < 2:
            time.sleep(0.5 * (attempt + 1))
    raise ValueError(f"Could not read frame count from {path}")


def _episode_video_error(dataset_dir: pathlib.Path, episode_index: int) -> str | None:
    try:
        for video_key in VIDEO_KEYS:
            _video_frame_count(_video_path(dataset_dir, episode_index, video_key))
    except (OSError, ValueError) as error:
        return str(error)
    return None


def _subsample_decodable_episodes(
    dataset_dir: pathlib.Path,
    episodes: list[int],
    max_episodes: int,
    *,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    initial = _subsample_episodes(episodes, max_episodes, seed=seed)
    target_count = len(initial)
    selected: list[int] = []
    rejected: list[dict[str, Any]] = []
    checked: set[int] = set()

    def consider(episode_index: int) -> None:
        checked.add(episode_index)
        error = _episode_video_error(dataset_dir, episode_index)
        if error is None:
            selected.append(episode_index)
        else:
            rejected.append({"episode_index": episode_index, "reason": error})

    for episode_index in initial:
        consider(episode_index)

    if len(selected) < target_count:
        remaining = np.asarray([episode for episode in episodes if episode not in checked], dtype=np.int64)
        replacement_order = np.random.default_rng(seed + 10_000).permutation(remaining)
        for episode_value in replacement_order:
            consider(int(episode_value))
            if len(selected) >= target_count:
                break

    if len(selected) != target_count:
        raise RuntimeError(
            f"Not enough decodable episodes: requested={target_count}, selected={len(selected)}, rejected={rejected}"
        )
    return sorted(selected), rejected


def _safe_episode_length(dataset_dir: pathlib.Path, episode_index: int, parquet_length: int) -> int:
    video_counts = [_video_frame_count(_video_path(dataset_dir, episode_index, video_key)) for video_key in VIDEO_KEYS]
    return min(parquet_length, *video_counts)


def _critical_offsets(
    dataset_dir: pathlib.Path,
    episode_index: int,
    *,
    count: int,
    min_separation: int,
    max_length: int,
) -> list[int]:
    if count <= 0:
        return []
    frame = pd.read_parquet(_episode_parquet_path(dataset_dir, episode_index), columns=["action", "observation.state"])
    actions = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in frame["action"].to_numpy()])
    states = np.stack(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in frame["observation.state"].to_numpy()]
    )
    safe_length = min(max_length, len(actions))
    actions = actions[:safe_length]
    states = states[:safe_length]
    if safe_length <= 0:
        return []

    joint_mask = np.ones(actions.shape[1], dtype=bool)
    for dim in (7, 15):
        if dim < joint_mask.size:
            joint_mask[dim] = False

    action_delta = np.zeros_like(actions)
    action_delta[1:] = actions[1:] - actions[:-1]
    action_score = np.linalg.norm(action_delta[:, joint_mask], axis=1)
    gripper_score = np.zeros(len(actions), dtype=np.float32)
    for dim in (7, 15):
        if dim < actions.shape[1]:
            gripper_score[1:] += np.abs(action_delta[1:, dim]).astype(np.float32)
        if dim < states.shape[1]:
            gripper_score[1:] += np.abs(states[1:, dim] - states[:-1, dim]).astype(np.float32)

    def normalize(values: np.ndarray) -> np.ndarray:
        spread = float(values.max() - values.min())
        if spread < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - values.min()) / spread).astype(np.float32)

    score = normalize(action_score) + normalize(gripper_score)
    order = np.argsort(-score)
    selected: list[int] = []
    for offset_value in order:
        offset = int(offset_value)
        if all(abs(offset - prev) >= min_separation for prev in selected):
            selected.append(offset)
            if len(selected) >= count:
                break
    return sorted(selected)


def _load_episode_actions(dataset_dir: pathlib.Path, episode_index: int) -> np.ndarray:
    frame = pd.read_parquet(_episode_parquet_path(dataset_dir, episode_index), columns=["action"])
    return np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in frame["action"].to_numpy()])


def _action_chunk(
    dataset_dir: pathlib.Path,
    episode_index: int,
    frame_offset: int,
    action_horizon: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    if episode_index not in cache:
        cache[episode_index] = _load_episode_actions(dataset_dir, episode_index)
    actions = cache[episode_index]
    end = min(frame_offset + action_horizon, len(actions))
    chunk = actions[frame_offset:end]
    if len(chunk) < action_horizon:
        pad = np.repeat(actions[-1][None, :], action_horizon - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32, copy=False)


def _make_samples(
    dataset: lerobot_dataset.LeRobotDataset,
    dataset_dir: pathlib.Path,
    episodes: list[int],
    *,
    uniform_frames: int,
    critical_frames: int,
    include_adjacent: bool,
) -> list[SampleSpec]:
    ep_to_arr = {ep: arr_idx for arr_idx, ep in enumerate(episodes)}
    starts = dataset.episode_data_index["from"].numpy()
    ends = dataset.episode_data_index["to"].numpy()
    samples: dict[tuple[int, int], SampleSpec] = {}

    for episode_index in episodes:
        arr_idx = ep_to_arr[episode_index]
        start = int(starts[arr_idx])
        end = int(ends[arr_idx])
        length = _safe_episode_length(dataset_dir, episode_index, end - start)
        if length <= 0:
            continue

        offsets: list[tuple[int, str]] = []
        if uniform_frames > 0:
            uniform = np.linspace(0, length - 1, num=min(uniform_frames, length), dtype=int)
            offsets.extend((int(offset), "uniform") for offset in uniform)

        min_sep = max(1, length // max(critical_frames * 3, 1))
        offsets.extend(
            (offset, "critical")
            for offset in _critical_offsets(
                dataset_dir,
                episode_index,
                count=min(critical_frames, length),
                min_separation=min_sep,
                max_length=length,
            )
        )

        if include_adjacent:
            offsets.extend((min(offset + 1, length - 1), f"{kind}_adjacent") for offset, kind in list(offsets))

        for offset, kind in offsets:
            key = (episode_index, offset)
            if key not in samples or (samples[key].kind != "critical" and kind == "critical"):
                samples[key] = SampleSpec(episode_index, offset, start + offset, kind)

    return sorted(samples.values(), key=lambda item: (item.episode_index, item.frame_offset))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_hwc_uint8(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    return image


def _build_observation(item: dict[str, Any], tasks: dict[int, str], prompt_override: str | None) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "observation.state": _to_numpy(item["observation.state"]).astype(np.float32),
    }
    for key in (
        "observation.images.base",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ):
        if key in item:
            obs[key] = _image_to_hwc_uint8(item[key])
    task_idx = int(_to_numpy(item.get("task_index", 0)).reshape(-1)[0])
    obs["prompt"] = prompt_override or tasks.get(task_idx, "fold the cloth")
    return obs


def _empty_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "mae_sum": 0.0,
        "rmse_sum": 0.0,
        "first_mae_sum": 0.0,
        "joint_mae_sum": 0.0,
        "gripper_mae_sum": 0.0,
        "overlap_sum": 0.0,
        "overlap_count": 0,
    }


def _add_metrics(
    acc: dict[str, Any],
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    prev_pred: np.ndarray | None,
    prev_index: int | None,
    index: int,
) -> None:
    horizon = min(len(pred), len(truth))
    action_dim = min(pred.shape[-1], truth.shape[-1])
    pred = pred[:horizon, :action_dim]
    truth = truth[:horizon, :action_dim]
    diff = pred - truth
    abs_diff = np.abs(diff)

    joint_mask = np.ones(action_dim, dtype=bool)
    gripper_mask = np.zeros(action_dim, dtype=bool)
    for dim in (7, 15):
        if dim < action_dim:
            joint_mask[dim] = False
            gripper_mask[dim] = True

    acc["count"] += 1
    acc["mae_sum"] += float(np.mean(abs_diff))
    acc["rmse_sum"] += float(np.sqrt(np.mean(diff**2)))
    acc["first_mae_sum"] += float(np.mean(abs_diff[0]))
    acc["joint_mae_sum"] += float(np.mean(abs_diff[:, joint_mask])) if np.any(joint_mask) else 0.0
    acc["gripper_mae_sum"] += float(np.mean(abs_diff[:, gripper_mask])) if np.any(gripper_mask) else 0.0

    if prev_pred is not None and prev_index is not None and index == prev_index + 1:
        overlap_horizon = min(len(prev_pred) - 1, len(pred), len(truth) - 1)
        if overlap_horizon > 0:
            overlap = np.abs(prev_pred[1 : 1 + overlap_horizon, :action_dim] - pred[:overlap_horizon, :action_dim])
            acc["overlap_sum"] += float(np.mean(overlap))
            acc["overlap_count"] += 1


def _finalize(acc: dict[str, Any]) -> dict[str, float | int]:
    count = max(int(acc["count"]), 1)
    overlap_count = max(int(acc["overlap_count"]), 1)
    return {
        "samples": int(acc["count"]),
        "mae": acc["mae_sum"] / count,
        "rmse": acc["rmse_sum"] / count,
        "first_step_mae": acc["first_mae_sum"] / count,
        "joint_mae": acc["joint_mae_sum"] / count,
        "gripper_mae": acc["gripper_mae_sum"] / count,
        "overlap_consistency_mae": acc["overlap_sum"] / overlap_count if acc["overlap_count"] else 0.0,
        "overlap_pairs": int(acc["overlap_count"]),
    }


def _evaluate_split(
    policy: Any,
    dataset: lerobot_dataset.LeRobotDataset,
    dataset_dir: pathlib.Path,
    tasks: dict[int, str],
    samples: list[SampleSpec],
    action_horizon: int,
    prompt_override: str | None,
) -> dict[str, Any]:
    by_kind = defaultdict(_empty_accumulator)
    overall = _empty_accumulator()
    per_episode = defaultdict(_empty_accumulator)
    prev_pred_by_episode: dict[int, tuple[int, np.ndarray]] = {}
    action_cache: dict[int, np.ndarray] = {}

    for sample in samples:
        item = dataset[sample.dataset_index]
        obs = _build_observation(item, tasks, prompt_override)
        pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
        truth = _action_chunk(
            dataset_dir,
            sample.episode_index,
            sample.frame_offset,
            action_horizon,
            action_cache,
        )

        prev_index = None
        prev_pred = None
        if sample.episode_index in prev_pred_by_episode:
            prev_index, prev_pred = prev_pred_by_episode[sample.episode_index]

        for acc in (overall, by_kind[sample.kind], per_episode[sample.episode_index]):
            _add_metrics(acc, pred, truth, prev_pred=prev_pred, prev_index=prev_index, index=sample.frame_offset)

        prev_pred_by_episode[sample.episode_index] = (sample.frame_offset, pred)

    episode_metrics = {str(ep): _finalize(acc) for ep, acc in sorted(per_episode.items(), key=lambda item: item[0])}
    worst = sorted(
        ((int(ep), values["mae"]) for ep, values in episode_metrics.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    return {
        "overall": _finalize(overall),
        "by_kind": {kind: _finalize(acc) for kind, acc in sorted(by_kind.items())},
        "episodes": episode_metrics,
        "worst_episodes_by_mae": [{"episode": ep, "mae": mae} for ep, mae in worst],
    }


def _checkpoint_step(checkpoint_dir: pathlib.Path) -> int:
    match = re.search(r"(\d+)$", checkpoint_dir.name)
    if not match:
        return -1
    return int(match.group(1))


def _load_cached_report(
    report_path: pathlib.Path,
    *,
    checkpoint_dir: pathlib.Path,
    config_name: str,
    dataset_dir: pathlib.Path,
    train_episodes: list[int],
    val_episodes: list[int],
    sampling: dict[str, Any],
) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    try:
        report = _load_json(report_path)
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "checkpoint": checkpoint_dir.resolve(),
        "config": config_name,
        "dataset": dataset_dir.resolve(),
        "step": _checkpoint_step(checkpoint_dir),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "sampling": sampling,
        "schema_version": REPORT_SCHEMA_VERSION,
    }
    try:
        actual = {
            "checkpoint": pathlib.Path(report["checkpoint"]).resolve(),
            "config": report["config"],
            "dataset": pathlib.Path(report["dataset"]).resolve(),
            "step": int(report["step"]),
            "train_episodes": report["train_episodes"],
            "val_episodes": report["val_episodes"],
            "sampling": report["sampling"],
            "schema_version": report["schema_version"],
        }
    except (KeyError, TypeError, ValueError):
        return None
    return report if actual == expected else None


def _sampling_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "critical_selector": CRITICAL_SELECTOR_VERSION,
        "uniform_frames": args.uniform_frames,
        "critical_frames": args.critical_frames,
        "include_adjacent": args.include_adjacent,
        "train_max_episodes": args.train_max_episodes,
        "val_max_episodes": args.val_max_episodes,
        "prompt_override": args.prompt,
    }


def _evaluate_checkpoint(
    *,
    config_name: str,
    checkpoint_dir: pathlib.Path,
    dataset_dir: pathlib.Path,
    train_episodes: list[int],
    val_episodes: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_config = _config.get_config(config_name)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(str(dataset_dir))

    LOGGER.info("Loading policy: %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

    def make_dataset(episodes: list[int]) -> lerobot_dataset.LeRobotDataset:
        return lerobot_dataset.LeRobotDataset(
            str(dataset_dir),
            episodes=episodes,
            tolerance_s=args.lerobot_tolerance_s,
        )

    LOGGER.info("Loading train sample dataset: %s episodes", len(train_episodes))
    train_dataset = make_dataset(train_episodes)
    train_samples = _make_samples(
        train_dataset,
        dataset_dir,
        train_episodes,
        uniform_frames=args.uniform_frames,
        critical_frames=args.critical_frames,
        include_adjacent=args.include_adjacent,
    )

    LOGGER.info("Loading val dataset: %s episodes", len(val_episodes))
    val_dataset = make_dataset(val_episodes)
    val_samples = _make_samples(
        val_dataset,
        dataset_dir,
        val_episodes,
        uniform_frames=args.uniform_frames,
        critical_frames=args.critical_frames,
        include_adjacent=args.include_adjacent,
    )

    LOGGER.info("Evaluating train samples: %s", len(train_samples))
    train_result = _evaluate_split(
        policy,
        train_dataset,
        dataset_dir,
        dataset_meta.tasks,
        train_samples,
        train_config.model.action_horizon,
        args.prompt,
    )
    LOGGER.info("Evaluating val samples: %s", len(val_samples))
    val_result = _evaluate_split(
        policy,
        val_dataset,
        dataset_dir,
        dataset_meta.tasks,
        val_samples,
        train_config.model.action_horizon,
        args.prompt,
    )

    train_mae = float(train_result["overall"]["mae"])
    val_mae = float(val_result["overall"]["mae"])
    train_critical = float(train_result["by_kind"].get("critical", {}).get("mae", 0.0))
    val_critical = float(val_result["by_kind"].get("critical", {}).get("mae", 0.0))

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "checkpoint": str(checkpoint_dir),
        "step": _checkpoint_step(checkpoint_dir),
        "config": config_name,
        "dataset": str(dataset_dir),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "sampling": _sampling_signature(args),
        "train": train_result,
        "val": val_result,
        "gaps": {
            "mae_val_minus_train": val_mae - train_mae,
            "mae_val_over_train": val_mae / train_mae if train_mae > 0 else 0.0,
            "critical_mae_val_minus_train": val_critical - train_critical,
            "critical_mae_val_over_train": val_critical / train_critical if train_critical > 0 else 0.0,
        },
    }


def _write_summary_csv(path: pathlib.Path, reports: Iterable[dict[str, Any]]) -> None:
    rows = [
        {
            "step": report["step"],
            "checkpoint": report["checkpoint"],
            "train_mae": report["train"]["overall"]["mae"],
            "val_mae": report["val"]["overall"]["mae"],
            "gap_mae": report["gaps"]["mae_val_minus_train"],
            "gap_ratio": report["gaps"]["mae_val_over_train"],
            "train_critical_mae": report["train"]["by_kind"].get("critical", {}).get("mae", 0.0),
            "val_critical_mae": report["val"]["by_kind"].get("critical", {}).get("mae", 0.0),
            "gap_critical_mae": report["gaps"]["critical_mae_val_minus_train"],
            "val_overlap_mae": report["val"]["overall"]["overlap_consistency_mae"],
            "val_gripper_mae": report["val"]["overall"]["gripper_mae"],
            "val_joint_mae": report["val"]["overall"]["joint_mae"],
            "val_samples": report["val"]["overall"]["samples"],
            "train_samples": report["train"]["overall"]["samples"],
        }
        for report in reports
    ]
    rows.sort(key=lambda row: int(row["step"]))
    fieldnames = list(rows[0].keys()) if rows else []
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_openarms_dual_hq")
    parser.add_argument(
        "--dataset", type=pathlib.Path, default=pathlib.Path("/share/home/linyongjia/datasets/high_quality_folding")
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--train-split", default="0:999")
    parser.add_argument("--val-split", default="999:1199")
    parser.add_argument("--train-max-episodes", type=int, default=80)
    parser.add_argument("--val-max-episodes", type=int, default=200)
    parser.add_argument("--uniform-frames", type=int, default=4)
    parser.add_argument("--critical-frames", type=int, default=4)
    parser.add_argument("--include-adjacent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--lerobot-tolerance-s", type=float, default=0.05)
    parser.add_argument("--prompt", default=None, help="Override every sampled frame prompt, e.g. AWBC positive mode.")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse complete per-checkpoint reports that match inputs."
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    train_episodes, train_rejected = _subsample_decodable_episodes(
        args.dataset,
        _parse_range(args.train_split),
        args.train_max_episodes,
        seed=args.seed,
    )
    val_episodes, val_rejected = _subsample_decodable_episodes(
        args.dataset,
        _parse_range(args.val_split),
        args.val_max_episodes,
        seed=args.seed + 1,
    )
    _write_json_atomic(
        args.output / "episode_selection.json",
        {
            "schema_version": "openarm_checkpoint_sweep_episode_selection_v1",
            "train": {"selected": train_episodes, "rejected": train_rejected},
            "val": {"selected": val_episodes, "rejected": val_rejected},
        },
    )

    reports = []
    sampling = _sampling_signature(args)
    for checkpoint_dir in args.checkpoint:
        report_path = args.output / f"checkpoint_{_checkpoint_step(checkpoint_dir)}.json"
        report = (
            _load_cached_report(
                report_path,
                checkpoint_dir=checkpoint_dir,
                config_name=args.config,
                dataset_dir=args.dataset,
                train_episodes=train_episodes,
                val_episodes=val_episodes,
                sampling=sampling,
            )
            if args.resume
            else None
        )
        if report is None:
            report = _evaluate_checkpoint(
                config_name=args.config,
                checkpoint_dir=checkpoint_dir,
                dataset_dir=args.dataset,
                train_episodes=train_episodes,
                val_episodes=val_episodes,
                args=args,
            )
            _write_json_atomic(report_path, report)
            LOGGER.info("Wrote %s", report_path)
        else:
            LOGGER.info("Reusing complete checkpoint report: %s", report_path)
        reports.append(report)
        gc.collect()

    summary_path = args.output / "summary.csv"
    _write_summary_csv(summary_path, reports)
    LOGGER.info("Wrote %s", summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()
