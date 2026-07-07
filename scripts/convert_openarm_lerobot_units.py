#!/usr/bin/env python3
"""Convert OpenArm LeRobot v2.1 state/action units without re-encoding videos."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil

import numpy as np
import pandas as pd

try:
    from scripts.write_lerobot_episode_stats import compute_episode_stats
except ModuleNotFoundError:
    from write_lerobot_episode_stats import compute_episode_stats

STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEYS = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.base",
)
GRIPPER_INDICES = (7, 15)
JOINT_INDICES = tuple(index for index in range(16) if index not in GRIPPER_INDICES)
DEFAULT_TASK = "Fold the T-shirt properly"
GRIPPER_DATASET_MAX_DEG = 66.0
GRIPPER_RAW_CLOSED_NORM = 0.0
GRIPPER_RAW_OPEN_NORM = 0.84


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _joint_unit_hint(max_abs_joint: float) -> str:
    if max_abs_joint <= np.pi + 0.25:
        return "radian_like"
    if max_abs_joint <= 360.0 + 1e-3:
        return "degree_like"
    return "large_or_mixed"


def _convert_policy_units(
    state: np.ndarray,
    action: np.ndarray,
    *,
    policy_joint_unit: str,
    policy_gripper_unit: str,
    gripper_closed_norm: float,
    gripper_open_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    converted_state = state.astype(np.float32, copy=True)
    converted_action = action.astype(np.float32, copy=True)

    if policy_joint_unit == "degrees":
        converted_state[:, JOINT_INDICES] *= np.float32(180.0 / np.pi)
        converted_action[:, JOINT_INDICES] *= np.float32(180.0 / np.pi)
    elif policy_joint_unit != "radians":
        raise ValueError(f"Unsupported policy joint unit: {policy_joint_unit}")

    if policy_gripper_unit == "dataset_degrees":
        if gripper_open_norm <= gripper_closed_norm:
            raise ValueError(
                f"gripper_open_norm must be > gripper_closed_norm, got {gripper_open_norm} <= {gripper_closed_norm}"
            )
        for array in (converted_state, converted_action):
            gripper = (array[:, GRIPPER_INDICES] - gripper_closed_norm) / (gripper_open_norm - gripper_closed_norm)
            gripper = np.clip(gripper, 0.0, 1.0)
            array[:, GRIPPER_INDICES] = -(1.0 - gripper) * np.float32(GRIPPER_DATASET_MAX_DEG)
    elif policy_gripper_unit != "normalized":
        raise ValueError(f"Unsupported policy gripper unit: {policy_gripper_unit}")

    return converted_state, converted_action


def _copy_or_link(src: pathlib.Path, dst: pathlib.Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def _episode_path(root: pathlib.Path, pattern: str, episode_index: int, chunks_size: int) -> pathlib.Path:
    return root / pattern.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index)


def _convert_episode(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    episode_index: int,
    chunks_size: int,
    data_path: str,
    policy_joint_unit: str,
    policy_gripper_unit: str,
    gripper_closed_norm: float,
    gripper_open_norm: float,
) -> dict:
    src_parquet = _episode_path(src, data_path, episode_index, chunks_size)
    frame = pd.read_parquet(src_parquet)
    state = np.stack(frame[STATE_KEY].map(np.asarray).to_numpy()).astype(np.float32)
    action = np.stack(frame[ACTION_KEY].map(np.asarray).to_numpy()).astype(np.float32)
    state, action = _convert_policy_units(
        state,
        action,
        policy_joint_unit=policy_joint_unit,
        policy_gripper_unit=policy_gripper_unit,
        gripper_closed_norm=gripper_closed_norm,
        gripper_open_norm=gripper_open_norm,
    )
    frame[STATE_KEY] = [row.copy() for row in state]
    frame[ACTION_KEY] = [row.copy() for row in action]

    dst_parquet = _episode_path(dst, data_path, episode_index, chunks_size)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dst_parquet, index=False)
    return compute_episode_stats(frame)


def convert_dataset(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    dataset_id: str,
    policy_joint_unit: str,
    policy_gripper_unit: str,
    task: str | None,
    gripper_closed_norm: float,
    gripper_open_norm: float,
    copy_mode: str,
    overwrite: bool,
) -> dict:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    info = _load_json(src / "meta/info.json")
    episodes = _read_jsonl(src / "meta/episodes.jsonl")
    tasks = _read_jsonl(src / "meta/tasks.jsonl")
    if task is not None:
        tasks = [{"task_index": 0, "task": task}]
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = str(info["data_path"])
    video_path = str(info["video_path"])

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    stats_rows: list[dict] = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        stats = _convert_episode(
            src,
            dst,
            episode_index=episode_index,
            chunks_size=chunks_size,
            data_path=data_path,
            policy_joint_unit=policy_joint_unit,
            policy_gripper_unit=policy_gripper_unit,
            gripper_closed_norm=gripper_closed_norm,
            gripper_open_norm=gripper_open_norm,
        )
        stats_rows.append({"episode_index": episode_index, "stats": stats})

        for video_key in VIDEO_KEYS:
            rel = pathlib.Path(
                video_path.format(
                    episode_chunk=episode_index // chunks_size,
                    video_key=video_key,
                    episode_index=episode_index,
                )
            )
            _copy_or_link(src / rel, dst / rel, copy_mode)

    out_info = dict(info)
    out_info["repo_id"] = dataset_id
    out_info["unit_conversion"] = {
        "source_dataset": str(src),
        "policy_joint_unit": policy_joint_unit,
        "policy_gripper_unit": policy_gripper_unit,
        "gripper_dataset_max_deg": GRIPPER_DATASET_MAX_DEG,
        "gripper_raw_closed_norm": gripper_closed_norm,
        "gripper_raw_open_norm": gripper_open_norm,
        "task": task,
    }
    _write_json(dst / "meta/info.json", out_info)
    _write_jsonl(dst / "meta/tasks.jsonl", tasks)
    if task is not None:
        episodes = [dict(row, tasks=[task]) for row in episodes]
    _write_jsonl(dst / "meta/episodes.jsonl", episodes)
    _write_jsonl(dst / "meta/episodes_stats.jsonl", stats_rows)

    max_abs_joint = 0.0
    max_abs_state_action = 0.0
    for row in stats_rows:
        for key in (STATE_KEY, ACTION_KEY):
            stats = row["stats"][key]
            mins = np.asarray(stats["min"], dtype=np.float32)
            maxs = np.asarray(stats["max"], dtype=np.float32)
            max_abs = np.maximum(np.abs(mins), np.abs(maxs))
            max_abs_joint = max(max_abs_joint, float(max_abs[list(JOINT_INDICES)].max()))
            max_abs_state_action = max(max_abs_state_action, float(max_abs.max()))

    report = {
        "source": str(src),
        "destination": str(dst),
        "dataset_id": dataset_id,
        "episodes": len(episodes),
        "policy_joint_unit": policy_joint_unit,
        "policy_gripper_unit": policy_gripper_unit,
        "task": task,
        "gripper_raw_closed_norm": gripper_closed_norm,
        "gripper_raw_open_norm": gripper_open_norm,
        "joint_unit_hint": _joint_unit_hint(max_abs_joint),
        "max_abs_joint": max_abs_joint,
        "max_abs_state_action": max_abs_state_action,
        "copy_mode": copy_mode,
    }
    _write_json(dst / "unit_conversion_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=pathlib.Path, required=True)
    parser.add_argument("--dst", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-id", default="openarm_site_align_v1_deg")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--policy-joint-unit", choices=("degrees", "radians"), default="degrees")
    parser.add_argument("--policy-gripper-unit", choices=("dataset_degrees", "normalized"), default="dataset_degrees")
    parser.add_argument("--gripper-closed-norm", type=float, default=GRIPPER_RAW_CLOSED_NORM)
    parser.add_argument("--gripper-open-norm", type=float, default=GRIPPER_RAW_OPEN_NORM)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_dataset(
        args.src,
        args.dst,
        dataset_id=args.dataset_id,
        policy_joint_unit=args.policy_joint_unit,
        policy_gripper_unit=args.policy_gripper_unit,
        task=args.task,
        gripper_closed_norm=args.gripper_closed_norm,
        gripper_open_norm=args.gripper_open_norm,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
