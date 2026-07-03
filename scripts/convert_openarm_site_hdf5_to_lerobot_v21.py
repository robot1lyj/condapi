"""Convert OpenArm on-site raw HDF5 episodes to LeRobot v2.1.

The OpenArm HIL/on-site recorder stores one HDF5 file per episode plus three
MP4 videos. This script keeps the videos as-is and writes only the tabular
state/action stream into LeRobot v2.1 parquet files, so conversion is fast and
does not re-encode camera data.

Example:
    python scripts/convert_openarm_site_hdf5_to_lerobot_v21.py \
      --src /storage1t/ipc \
      --dst /storage1t/datasets/openarm_site_align_v1 \
      --dataset-id openarm_site_align_v1 \
      --task "fold the cloth" \
      --val-count 10 \
      --copy-mode hardlink \
      --overwrite
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import dataclasses
import json
import math
import os
import pathlib
import re
import shutil
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd

STATE_KEY = "observation.state"
ACTION_KEY = "action"
VIDEO_KEYS = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.base",
)
GRIPPER_INDICES = (7, 15)
JOINT_INDICES = tuple(index for index in range(16) if index not in GRIPPER_INDICES)
JOINT_NAMES = [
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_joint_7.pos",
    "right_gripper.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_joint_7.pos",
    "left_gripper.pos",
]


@dataclasses.dataclass(frozen=True)
class RawSource:
    name: str
    path: pathlib.Path
    info: dict[str, Any]
    episodes_meta: dict[int, dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class EpisodeCandidate:
    source: RawSource
    hdf5_path: pathlib.Path
    source_episode_index: int
    raw_meta: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class InspectedEpisode:
    candidate: EpisodeCandidate
    length: int
    video_paths: dict[str, pathlib.Path]
    max_abs_state_action: float
    max_abs_joint: float
    gripper_action_nan_count: int
    timestamp_start: float
    timestamp_end: float


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_episodes(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Episode spec is empty")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid episode range: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, end, step))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _episode_index_from_path(path: pathlib.Path) -> int:
    match = re.search(r"episode_(\d+)\.hdf5$", path.name)
    if not match:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def _discover_raw_sources(src_paths: list[pathlib.Path]) -> list[RawSource]:
    roots: list[pathlib.Path] = []
    for src_path in src_paths:
        src_root = src_path.expanduser().resolve()
        if (src_root / "meta/info.json").exists() and (src_root / "episodes").is_dir():
            roots.append(src_root)
            continue
        if not src_root.exists():
            raise FileNotFoundError(src_root)
        roots.extend(
            child
            for child in sorted(src_root.iterdir())
            if child.is_dir() and (child / "meta/info.json").exists() and (child / "episodes").is_dir()
        )

    seen: set[pathlib.Path] = set()
    sources: list[RawSource] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        info = _load_json(root / "meta/info.json")
        episodes_meta = {
            int(row["episode_index"]): row
            for row in _load_jsonl(root / "meta/episodes.jsonl")
            if "episode_index" in row
        }
        sources.append(RawSource(name=root.name, path=root, info=info, episodes_meta=episodes_meta))
    if not sources:
        raise FileNotFoundError(f"No raw OpenArm HDF5 sources found under {src_paths}")
    return sources


def _collect_candidates(sources: list[RawSource]) -> list[EpisodeCandidate]:
    candidates: list[EpisodeCandidate] = []
    for source in sorted(sources, key=lambda item: item.name):
        hdf5_paths = sorted(source.path.glob("episodes/*.hdf5"), key=_episode_index_from_path)
        for hdf5_path in hdf5_paths:
            source_episode_index = _episode_index_from_path(hdf5_path)
            raw_meta = dict(source.episodes_meta.get(source_episode_index, {}))
            candidates.append(
                EpisodeCandidate(
                    source=source,
                    hdf5_path=hdf5_path,
                    source_episode_index=source_episode_index,
                    raw_meta=raw_meta,
                )
            )
    return candidates


def _select_candidates(
    candidates: list[EpisodeCandidate],
    *,
    episodes: str | None,
    max_episodes: int | None,
) -> list[EpisodeCandidate]:
    if episodes is not None:
        selected_indices = parse_episodes(episodes)
        try:
            candidates = [candidates[index] for index in selected_indices]
        except IndexError as exc:
            raise ValueError(f"Episode selection {episodes!r} exceeds {len(candidates)} collected episodes") from exc
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError(f"--max-episodes must be positive, got {max_episodes}")
        candidates = candidates[:max_episodes]
    return candidates


def _require_dataset(file: h5py.File, key: str) -> h5py.Dataset:
    if key not in file:
        raise KeyError(f"Missing HDF5 dataset {key!r}")
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"HDF5 key {key!r} is not a dataset")
    return dataset


def _video_path_from_hdf5(file: h5py.File, candidate: EpisodeCandidate, video_key: str) -> pathlib.Path:
    group_key = f"videos/{video_key}"
    if group_key not in file:
        raise KeyError(f"Missing HDF5 video group {group_key!r}")
    video_group = file[group_key]
    raw_path = video_group.attrs.get("path")
    if raw_path is None:
        raw_videos = candidate.raw_meta.get("videos", {})
        raw_path = raw_videos.get(video_key)
    if raw_path is None:
        raise KeyError(f"Missing video path for {video_key!r}")
    return candidate.source.path / str(raw_path)


def _joint_unit_hint(max_abs_joint: float) -> str:
    if max_abs_joint <= math.pi + 0.25:
        return "radian_like"
    if max_abs_joint <= 360.0 + 1e-3:
        return "degree_like"
    return "large_or_mixed"


def _fill_gripper_action(action: np.ndarray, state: np.ndarray, mode: str) -> np.ndarray:
    clean_action = action.astype(np.float32, copy=True)
    gripper_values = clean_action[:, GRIPPER_INDICES]
    missing = ~np.isfinite(gripper_values)
    if not missing.any():
        return clean_action
    if mode == "state":
        fallback = state[:, GRIPPER_INDICES]
    elif mode == "zero":
        fallback = np.zeros_like(gripper_values)
    elif mode == "error":
        return clean_action
    else:
        raise ValueError(f"Unsupported gripper action fallback: {mode}")
    gripper_values[missing] = fallback[missing]
    clean_action[:, GRIPPER_INDICES] = gripper_values
    return clean_action


def inspect_episode(
    candidate: EpisodeCandidate,
    *,
    min_frames: int,
    max_abs_state_action: float | None,
    verify_video_frames: bool,
    gripper_action_fallback: str,
) -> InspectedEpisode:
    with h5py.File(candidate.hdf5_path, "r") as file:
        state_ds = _require_dataset(file, STATE_KEY)
        action_ds = _require_dataset(file, ACTION_KEY)
        timestamp_ds = _require_dataset(file, "timestamp")

        if state_ds.ndim != 2 or state_ds.shape[1] != 16:
            raise ValueError(f"{STATE_KEY} expected shape [T,16], got {state_ds.shape}")
        if action_ds.ndim != 2 or action_ds.shape[1] != 16:
            raise ValueError(f"{ACTION_KEY} expected shape [T,16], got {action_ds.shape}")
        length = int(state_ds.shape[0])
        if int(action_ds.shape[0]) != length or int(timestamp_ds.shape[0]) != length:
            raise ValueError(
                f"state/action/timestamp length mismatch: {state_ds.shape[0]}, {action_ds.shape[0]}, "
                f"{timestamp_ds.shape[0]}"
            )
        if length < min_frames:
            raise ValueError(f"Episode too short: {length} < {min_frames}")

        state = np.asarray(state_ds, dtype=np.float32)
        action = np.asarray(action_ds, dtype=np.float32)
        timestamp = np.asarray(timestamp_ds, dtype=np.float64)
        if not np.isfinite(state).all():
            raise ValueError(f"{STATE_KEY} contains NaN or Inf")
        if not np.isfinite(action[:, JOINT_INDICES]).all():
            raise ValueError(f"{ACTION_KEY} joint dimensions contain NaN or Inf")
        gripper_action_nan_count = int((~np.isfinite(action[:, GRIPPER_INDICES])).sum())
        if gripper_action_nan_count and gripper_action_fallback == "error":
            raise ValueError(f"{ACTION_KEY} gripper dimensions contain {gripper_action_nan_count} NaN/Inf values")
        if not np.isfinite(timestamp).all():
            raise ValueError("timestamp contains NaN or Inf")

        clean_action = _fill_gripper_action(action, state, gripper_action_fallback)
        if not np.isfinite(clean_action).all():
            raise ValueError(f"{ACTION_KEY} still contains NaN or Inf after gripper fallback")

        max_abs = float(max(np.max(np.abs(state)), np.max(np.abs(clean_action))))
        if max_abs_state_action is not None and max_abs > max_abs_state_action:
            raise ValueError(f"state/action max abs {max_abs:.6g} exceeds {max_abs_state_action:.6g}")

        max_abs_joint = float(
            max(np.max(np.abs(state[:, JOINT_INDICES])), np.max(np.abs(clean_action[:, JOINT_INDICES])))
        )

        video_paths = {}
        for video_key in VIDEO_KEYS:
            video_path = _video_path_from_hdf5(file, candidate, video_key)
            if not video_path.exists():
                raise FileNotFoundError(f"Missing video file for {video_key}: {video_path}")
            video_paths[video_key] = video_path
            raw_num_frames = file[f"videos/{video_key}"].attrs.get("num_frames")
            if raw_num_frames is not None and int(raw_num_frames) != length:
                raise ValueError(f"{video_key} HDF5 video frame attr {int(raw_num_frames)} != tabular length {length}")
            if verify_video_frames:
                actual_frames = _read_video_frame_count(video_path)
                if actual_frames is not None and actual_frames != length:
                    raise ValueError(f"{video_key} actual video frames {actual_frames} != tabular length {length}")

    return InspectedEpisode(
        candidate=candidate,
        length=length,
        video_paths=video_paths,
        max_abs_state_action=max_abs,
        max_abs_joint=max_abs_joint,
        gripper_action_nan_count=gripper_action_nan_count,
        timestamp_start=float(timestamp[0]),
        timestamp_end=float(timestamp[-1]),
    )


def _read_video_frame_count(path: pathlib.Path) -> int | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count if frame_count > 0 else None
    finally:
        capture.release()


def _inspect_candidates(
    candidates: list[EpisodeCandidate],
    *,
    min_frames: int,
    max_abs_state_action: float | None,
    verify_video_frames: bool,
    gripper_action_fallback: str,
) -> tuple[list[InspectedEpisode], list[dict[str, Any]]]:
    valid: list[InspectedEpisode] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            valid.append(
                inspect_episode(
                    candidate,
                    min_frames=min_frames,
                    max_abs_state_action=max_abs_state_action,
                    verify_video_frames=verify_video_frames,
                    gripper_action_fallback=gripper_action_fallback,
                )
            )
        except Exception as exc:
            rejected.append(
                {
                    "source_dataset": candidate.source.name,
                    "source_episode_index": candidate.source_episode_index,
                    "hdf5_path": str(candidate.hdf5_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return valid, rejected


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


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_data_path(episode_index: int, chunks_size: int) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, chunks_size)
    return pathlib.Path(f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet")


def _format_video_path(episode_index: int, chunks_size: int, video_key: str) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, chunks_size)
    return pathlib.Path(f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")


def _zeroed_timestamps(raw_timestamp: np.ndarray, *, fps: int, mode: str) -> np.ndarray:
    if mode == "fps":
        return (np.arange(len(raw_timestamp), dtype=np.float32) / np.float32(fps)).astype(np.float32)
    if mode != "raw_zeroed":
        raise ValueError(f"Unsupported timestamp mode: {mode}")
    timestamp = raw_timestamp.astype(np.float32, copy=True)
    timestamp -= timestamp[0]
    if np.any(np.diff(timestamp) < -1e-4):
        return (np.arange(len(raw_timestamp), dtype=np.float32) / np.float32(fps)).astype(np.float32)
    return timestamp


def _write_episode_parquet(
    inspected: InspectedEpisode,
    dst: pathlib.Path,
    *,
    episode_index: int,
    global_start_index: int,
    chunks_size: int,
    fps: int,
    timestamp_mode: str,
    gripper_action_fallback: str,
) -> pathlib.Path:
    with h5py.File(inspected.candidate.hdf5_path, "r") as file:
        state = np.asarray(file[STATE_KEY], dtype=np.float32)
        action = np.asarray(file[ACTION_KEY], dtype=np.float32)
        raw_timestamp = np.asarray(file["timestamp"], dtype=np.float64)

    length = inspected.length
    action = _fill_gripper_action(action, state, gripper_action_fallback)
    timestamp = _zeroed_timestamps(raw_timestamp, fps=fps, mode=timestamp_mode)
    frame = pd.DataFrame(
        {
            ACTION_KEY: [row.copy() for row in action],
            STATE_KEY: [row.copy() for row in state],
            "timestamp": timestamp,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_start_index, global_start_index + length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
        }
    )
    parquet_path = dst / _format_data_path(episode_index, chunks_size)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    return parquet_path


def _feature_schema(raw_infos: list[dict[str, Any]], *, fps: int, video_codec: str) -> dict[str, Any]:
    merged_raw_features: dict[str, Any] = {}
    for info in raw_infos:
        merged_raw_features.update(info.get("schema", {}).get("features", {}))

    def video_feature(video_key: str, fallback_shape: list[int]) -> dict[str, Any]:
        shape = list(merged_raw_features.get(video_key, {}).get("shape", fallback_shape))
        return {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": int(shape[0]),
                "video.width": int(shape[1]),
                "video.codec": video_codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": int(fps),
                "video.channels": int(shape[2]),
                "has_audio": False,
            },
        }

    return {
        ACTION_KEY: {"dtype": "float32", "shape": [16], "names": JOINT_NAMES},
        STATE_KEY: {"dtype": "float32", "shape": [16], "names": JOINT_NAMES},
        "observation.images.left_wrist": video_feature("observation.images.left_wrist", [720, 1280, 3]),
        "observation.images.right_wrist": video_feature("observation.images.right_wrist", [720, 1280, 3]),
        "observation.images.base": video_feature("observation.images.base", [480, 640, 3]),
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _make_info(
    sources: list[RawSource],
    inspected: list[InspectedEpisode],
    *,
    dataset_id: str,
    task: str,
    fps: int,
    chunks_size: int,
    val_count: int,
    video_codec: str,
) -> dict[str, Any]:
    total_episodes = len(inspected)
    total_frames = sum(item.length for item in inspected)
    total_videos = total_episodes * len(VIDEO_KEYS)
    train_end = max(0, total_episodes - val_count)
    splits = {"train": f"0:{train_end}"}
    if val_count > 0:
        splits["val"] = f"{train_end}:{total_episodes}"
    return {
        "codebase_version": "v2.1",
        "robot_type": "openarms_follower",
        "repo_id": dataset_id,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": max(1, math.ceil(total_episodes / chunks_size)),
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": splits,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _feature_schema([source.info for source in sources], fps=fps, video_codec=video_codec),
        "site_conversion": {
            "dataset_id": dataset_id,
            "task": task,
            "source_roots": [str(source.path) for source in sources],
            "raw_format_versions": sorted(
                {
                    str(source.info.get("raw_format_version"))
                    for source in sources
                    if source.info.get("raw_format_version") is not None
                }
            ),
        },
    }


def convert_dataset(
    src_paths: list[pathlib.Path],
    dst: pathlib.Path,
    *,
    dataset_id: str,
    task: str,
    val_count: int,
    fps: int,
    chunks_size: int,
    copy_mode: str,
    overwrite: bool,
    dry_run: bool,
    episodes: str | None,
    max_episodes: int | None,
    min_frames: int,
    max_abs_state_action: float | None,
    timestamp_mode: str,
    gripper_action_fallback: str,
    verify_video_frames: bool,
    skip_invalid: bool,
    site_repeat: int,
) -> dict[str, Any]:
    sources = _discover_raw_sources(src_paths)
    candidates = _select_candidates(_collect_candidates(sources), episodes=episodes, max_episodes=max_episodes)
    inspected, rejected = _inspect_candidates(
        candidates,
        min_frames=min_frames,
        max_abs_state_action=max_abs_state_action,
        verify_video_frames=verify_video_frames,
        gripper_action_fallback=gripper_action_fallback,
    )
    if rejected and not skip_invalid and not dry_run:
        _write_json(
            dst.with_name(f"{dst.name}_conversion_rejected_report.json"),
            {"destination": str(dst), "rejected": rejected},
        )
        raise ValueError(f"{len(rejected)} invalid episodes found; inspect rejection report or pass --skip-invalid")
    if val_count < 0:
        raise ValueError(f"--val-count must be >= 0, got {val_count}")
    if val_count >= len(inspected) and inspected:
        raise ValueError(f"--val-count {val_count} must be smaller than valid episode count {len(inspected)}")

    max_abs_joint = max((item.max_abs_joint for item in inspected), default=0.0)
    train_end = max(0, len(inspected) - val_count)
    report = {
        "destination": str(dst),
        "dataset_id": dataset_id,
        "task": task,
        "dry_run": dry_run,
        "selected_episodes": len(candidates),
        "valid_episodes": len(inspected),
        "rejected_episodes": len(rejected),
        "total_frames": sum(item.length for item in inspected),
        "fps": fps,
        "copy_mode": copy_mode,
        "timestamp_mode": timestamp_mode,
        "gripper_action_fallback": gripper_action_fallback,
        "splits": {"train": f"0:{train_end}", **({"val": f"{train_end}:{len(inspected)}"} if val_count else {})},
        "joint_unit_hint": _joint_unit_hint(max_abs_joint),
        "max_abs_joint": max_abs_joint,
        "max_abs_state_action": max((item.max_abs_state_action for item in inspected), default=0.0),
        "gripper_action_nan_count": sum(item.gripper_action_nan_count for item in inspected),
        "sources": [
            {
                "name": source.name,
                "path": str(source.path),
                "raw_total_episodes": source.info.get("total_episodes"),
                "raw_total_frames": source.info.get("total_frames"),
            }
            for source in sources
        ],
        "rejected": rejected,
        "recommended_merge_source": f"site,{dst},0:{train_end},{site_repeat}",
    }
    if dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return report

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    total_frames = 0
    episode_rows: list[dict[str, Any]] = []
    for new_episode_index, item in enumerate(inspected):
        _write_episode_parquet(
            item,
            dst,
            episode_index=new_episode_index,
            global_start_index=total_frames,
            chunks_size=chunks_size,
            fps=fps,
            timestamp_mode=timestamp_mode,
            gripper_action_fallback=gripper_action_fallback,
        )
        for video_key, src_video in item.video_paths.items():
            _copy_or_link(src_video, dst / _format_video_path(new_episode_index, chunks_size, video_key), copy_mode)

        raw_tasks = item.candidate.raw_meta.get("tasks", [])
        episode_rows.append(
            {
                "episode_index": new_episode_index,
                "tasks": [task],
                "length": item.length,
                "source_dataset": item.candidate.source.name,
                "source_episode_index": item.candidate.source_episode_index,
                "source_hdf5_path": str(item.candidate.hdf5_path),
                "raw_tasks": raw_tasks,
                "timestamp_start": item.timestamp_start,
                "timestamp_end": item.timestamp_end,
            }
        )
        total_frames += item.length

    video_codec = str(sources[0].info.get("vcodec") or "h264")
    info = _make_info(
        sources,
        inspected,
        dataset_id=dataset_id,
        task=task,
        fps=fps,
        chunks_size=chunks_size,
        val_count=val_count,
        video_codec=video_codec,
    )
    _write_json(dst / "meta/info.json", info)
    _write_jsonl(dst / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(dst / "meta/episodes.jsonl", episode_rows)
    _write_json(dst / "conversion_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=pathlib.Path,
        action="append",
        default=[],
        help="Raw root. Can be /storage1t/ipc or an individual fold_cloth* directory. Repeatable.",
    )
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="Destination LeRobot v2.1 dataset root.")
    parser.add_argument("--dataset-id", default="openarm_site_align_v1")
    parser.add_argument("--task", default="fold the cloth")
    parser.add_argument("--episodes", default=None, help="Flattened candidate spec, e.g. 0:100 or 0,2,5.")
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="Use first N flattened candidates after sorting."
    )
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--timestamp-mode", choices=("raw_zeroed", "fps"), default="raw_zeroed")
    parser.add_argument(
        "--gripper-action-fallback",
        choices=("state", "zero", "error"),
        default="state",
        help="How to fill NaN/Inf gripper action dimensions 7 and 15. Site raw data currently uses NaN there.",
    )
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-abs-state-action", type=float, default=None)
    parser.add_argument("--verify-video-frames", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--site-repeat", type=int, default=5, help="Only used in the recommended merge command.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    src_paths = args.src or [pathlib.Path("/storage1t/ipc")]
    convert_dataset(
        src_paths,
        args.dst,
        dataset_id=args.dataset_id,
        task=args.task,
        val_count=args.val_count,
        fps=args.fps,
        chunks_size=args.chunks_size,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        episodes=args.episodes,
        max_episodes=args.max_episodes,
        min_frames=args.min_frames,
        max_abs_state_action=args.max_abs_state_action,
        timestamp_mode=args.timestamp_mode,
        gripper_action_fallback=args.gripper_action_fallback,
        verify_video_frames=args.verify_video_frames,
        skip_invalid=args.skip_invalid,
        site_repeat=args.site_repeat,
    )


if __name__ == "__main__":
    main()
