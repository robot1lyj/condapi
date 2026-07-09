"""Convert OpenArm HIL raw HDF5 episodes to a clean LeRobot v2.1 dataset.

The HIL recorder keeps a raw debug stream with policy, hold, and human-control
frames. Evo-RL should only see clean policy frames and real human VR
intervention frames:

- keep policy frames: ``session_state=policy`` and ``authority_source=policy``
- drop hold/wait frames: ``session_state=intervention_hold`` or ``selected_source=hold``
- keep human frames: ``session_state=human`` and ``authority_source=human`` with finite teleop action

Videos are rewritten with the same frame filter so the LeRobot video stream
stays aligned with the filtered parquet rows.
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
DEFAULT_TASK = "Fold the T-shirt properly"
DEFAULT_DATASET_ID = "openarm_hil_evo_v1"
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
    raw_length: int
    keep_indices: np.ndarray
    human_mask_kept: np.ndarray
    policy_count: int
    human_count: int
    hold_count: int
    dropped_other_count: int
    video_paths: dict[str, pathlib.Path]
    timestamp_start: float
    timestamp_end: float

    @property
    def length(self) -> int:
        return int(self.keep_indices.shape[0])


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
        raise FileNotFoundError(f"No raw OpenArm HIL sources found under {src_paths}")
    return sources


def _collect_candidates(sources: list[RawSource]) -> list[EpisodeCandidate]:
    candidates: list[EpisodeCandidate] = []
    for source in sorted(sources, key=lambda item: item.name):
        for hdf5_path in sorted(source.path.glob("episodes/*.hdf5"), key=_episode_index_from_path):
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


def _first_existing_dataset(file: h5py.File, aliases: tuple[str, ...]) -> h5py.Dataset | None:
    for key in aliases:
        if key in file and isinstance(file[key], h5py.Dataset):
            return file[key]
    return None


def _read_array_feature(
    file: h5py.File,
    aliases: tuple[str, ...],
    *,
    length: int,
    shape: tuple[int, ...],
    required: bool = False,
    fill_value: float = np.nan,
) -> np.ndarray:
    dataset = _first_existing_dataset(file, aliases)
    if dataset is None:
        if required:
            raise KeyError(f"Missing any of HDF5 datasets {aliases!r}")
        return np.full((length, *shape), fill_value, dtype=np.float32)
    array = np.asarray(dataset, dtype=np.float32)
    if array.shape != (length, *shape):
        raise ValueError(f"{dataset.name} expected shape {(length, *shape)}, got {array.shape}")
    return array


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    return str(value)


def _read_string_feature(
    file: h5py.File,
    aliases: tuple[str, ...],
    *,
    length: int,
    default: str = "",
) -> list[str]:
    dataset = _first_existing_dataset(file, aliases)
    if dataset is None:
        return [default] * length
    if h5py.check_string_dtype(dataset.dtype) is not None:
        values = dataset.asstr()[()]
    else:
        values = dataset[()]
    values = np.asarray(values)
    if values.ndim == 0:
        return [_decode_string(values.item())] * length
    if values.shape[0] != length:
        raise ValueError(f"{dataset.name} length mismatch: {values.shape[0]} != {length}")
    return [_decode_string(value) for value in values.reshape(length)]


def _normalize_label(value: str) -> str:
    return str(value).strip().lower()


def _video_path_from_hdf5(file: h5py.File, candidate: EpisodeCandidate, video_key: str) -> pathlib.Path:
    group_key = f"videos/{video_key}"
    raw_path = None
    if group_key in file:
        video_group = file[group_key]
        raw_path = video_group.attrs.get("path")
    if raw_path is None:
        raw_path = dict(candidate.raw_meta.get("videos", {})).get(video_key)
    if raw_path is None:
        fallback = candidate.source.path / "videos" / video_key / f"episode_{candidate.source_episode_index:06d}.mp4"
        raw_path = fallback.relative_to(candidate.source.path) if fallback.exists() else None
    if raw_path is None:
        raise KeyError(f"Missing video path for {video_key!r}")
    return candidate.source.path / str(raw_path)


def _read_video_frame_count(path: pathlib.Path) -> int | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count if frame_count > 0 else None
    finally:
        capture.release()


def _read_video_shape(path: pathlib.Path) -> list[int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video {path}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Cannot read first frame from {path}")
        height, width = frame.shape[:2]
        channels = 1 if frame.ndim == 2 else int(frame.shape[2])
        return [int(height), int(width), channels]
    finally:
        capture.release()


def _episode_success_from_meta(raw_meta: dict[str, Any], default: str) -> str:
    value = raw_meta.get("episode_success")
    if value is None:
        outcome = raw_meta.get("episode_outcome")
        if outcome in {"success", "failure"}:
            value = outcome
    if value is None:
        value = default
    value = str(value).strip().lower()
    if value not in {"success", "failure"}:
        raise ValueError(f"episode_success must resolve to success/failure, got {value!r}")
    return value


def inspect_episode(
    candidate: EpisodeCandidate,
    *,
    min_frames: int,
    verify_video_frames: bool,
    default_success: str,
) -> InspectedEpisode:
    with h5py.File(candidate.hdf5_path, "r") as file:
        state_ds = _require_dataset(file, STATE_KEY)
        action_ds = _first_existing_dataset(file, ("action.executed", "action", "executed_action"))
        timestamp_ds = _require_dataset(file, "timestamp")
        if action_ds is None:
            raise KeyError("Missing action/action.executed/executed_action")

        if state_ds.ndim != 2 or state_ds.shape[1] != 16:
            raise ValueError(f"{STATE_KEY} expected shape [T,16], got {state_ds.shape}")
        if action_ds.ndim != 2 or action_ds.shape[1] != 16:
            raise ValueError(f"{action_ds.name} expected shape [T,16], got {action_ds.shape}")
        length = int(state_ds.shape[0])
        if int(action_ds.shape[0]) != length or int(timestamp_ds.shape[0]) != length:
            raise ValueError(
                f"state/action/timestamp length mismatch: {state_ds.shape[0]}, {action_ds.shape[0]}, "
                f"{timestamp_ds.shape[0]}"
            )

        state = np.asarray(state_ds, dtype=np.float32)
        action = np.asarray(action_ds, dtype=np.float32)
        timestamp = np.asarray(timestamp_ds, dtype=np.float64)
        teleop_action = _read_array_feature(
            file,
            ("teleop_action", "human_action", "action.human"),
            length=length,
            shape=(16,),
        )
        session_state = [
            _normalize_label(item) for item in _read_string_feature(file, ("session_state",), length=length)
        ]
        selected_source = [
            _normalize_label(item) for item in _read_string_feature(file, ("selected_source",), length=length)
        ]
        authority_source = [
            _normalize_label(item) for item in _read_string_feature(file, ("authority_source",), length=length)
        ]

        if not np.isfinite(state).all():
            raise ValueError(f"{STATE_KEY} contains NaN or Inf")
        if not np.isfinite(action).all():
            raise ValueError(f"{action_ds.name} contains NaN or Inf")
        if not np.isfinite(timestamp).all():
            raise ValueError("timestamp contains NaN or Inf")
        _episode_success_from_meta(candidate.raw_meta, default_success)

        teleop_finite = np.isfinite(teleop_action).all(axis=1)
        session_arr = np.asarray(session_state, dtype=object)
        selected_arr = np.asarray(selected_source, dtype=object)
        authority_arr = np.asarray(authority_source, dtype=object)

        policy_mask = (session_arr == "policy") & (authority_arr == "policy") & (selected_arr == "policy")
        hold_mask = (session_arr == "intervention_hold") | (selected_arr == "hold")
        human_mask = (session_arr == "human") & (authority_arr == "human") & (selected_arr == "human") & teleop_finite
        keep_mask = policy_mask | human_mask
        dropped_other_mask = ~keep_mask & ~hold_mask
        keep_indices = np.flatnonzero(keep_mask).astype(np.int64)
        if keep_indices.shape[0] < min_frames:
            raise ValueError(f"Clean HIL episode too short after dropping hold/invalid frames: {keep_indices.shape[0]}")

        video_paths = {}
        for video_key in VIDEO_KEYS:
            video_path = _video_path_from_hdf5(file, candidate, video_key)
            if not video_path.exists():
                raise FileNotFoundError(f"Missing video file for {video_key}: {video_path}")
            video_paths[video_key] = video_path
            raw_num_frames = None
            group_key = f"videos/{video_key}"
            if group_key in file:
                raw_num_frames = file[group_key].attrs.get("num_frames")
            if raw_num_frames is not None and int(raw_num_frames) != length:
                raise ValueError(f"{video_key} HDF5 video frame attr {int(raw_num_frames)} != tabular length {length}")
            if verify_video_frames:
                actual_frames = _read_video_frame_count(video_path)
                if actual_frames is not None and actual_frames != length:
                    raise ValueError(f"{video_key} actual video frames {actual_frames} != tabular length {length}")

    return InspectedEpisode(
        candidate=candidate,
        raw_length=length,
        keep_indices=keep_indices,
        human_mask_kept=human_mask[keep_indices].astype(np.bool_),
        policy_count=int(policy_mask.sum()),
        human_count=int(human_mask.sum()),
        hold_count=int(hold_mask.sum()),
        dropped_other_count=int(dropped_other_mask.sum()),
        video_paths=video_paths,
        timestamp_start=float(timestamp[0]),
        timestamp_end=float(timestamp[-1]),
    )


def _inspect_candidates(
    candidates: list[EpisodeCandidate],
    *,
    min_frames: int,
    verify_video_frames: bool,
    default_success: str,
) -> tuple[list[InspectedEpisode], list[dict[str, Any]]]:
    valid: list[InspectedEpisode] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            valid.append(
                inspect_episode(
                    candidate,
                    min_frames=min_frames,
                    verify_video_frames=verify_video_frames,
                    default_success=default_success,
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


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_data_path(episode_index: int, chunks_size: int) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, chunks_size)
    return pathlib.Path(f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet")


def _format_video_path(episode_index: int, chunks_size: int, video_key: str) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, chunks_size)
    return pathlib.Path(f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")


def _zeroed_timestamps(raw_timestamp: np.ndarray, keep_indices: np.ndarray, *, fps: float, mode: str) -> np.ndarray:
    if mode == "fps":
        return (np.arange(len(keep_indices), dtype=np.float32) / np.float32(fps)).astype(np.float32)
    if mode != "raw_zeroed":
        raise ValueError(f"Unsupported timestamp mode: {mode}")
    timestamp = raw_timestamp[keep_indices].astype(np.float32, copy=True)
    timestamp -= timestamp[0]
    if np.any(np.diff(timestamp) < -1e-4):
        return (np.arange(len(keep_indices), dtype=np.float32) / np.float32(fps)).astype(np.float32)
    return timestamp


def _write_filtered_video(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    keep_indices: np.ndarray,
    fps: int,
    codec: str,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video {src}")
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Cannot read first frame from {src}")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*codec), float(fps), (int(width), int(height)))
        if not writer.isOpened():
            raise ValueError(f"Cannot open video writer for {dst} with codec {codec!r}")
        try:
            keep_set = {int(index) for index in keep_indices.tolist()}
            frame_idx = 0
            while ok and frame is not None:
                if frame_idx in keep_set:
                    writer.write(frame)
                frame_idx += 1
                ok, frame = capture.read()
        finally:
            writer.release()
    finally:
        capture.release()

    actual_frames = _read_video_frame_count(dst)
    if actual_frames is not None and actual_frames != keep_indices.shape[0]:
        raise ValueError(f"Filtered video {dst} has {actual_frames} frames, expected {keep_indices.shape[0]}")


def _write_episode_parquet(
    inspected: InspectedEpisode,
    dst: pathlib.Path,
    *,
    episode_index: int,
    global_start_index: int,
    chunks_size: int,
    fps: int,
    timestamp_mode: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    keep = inspected.keep_indices
    with h5py.File(inspected.candidate.hdf5_path, "r") as file:
        state = np.asarray(file[STATE_KEY], dtype=np.float32)[keep]
        action_ds = _first_existing_dataset(file, ("action.executed", "action", "executed_action"))
        if action_ds is None:
            raise KeyError("Missing action/action.executed/executed_action")
        action = np.asarray(action_ds, dtype=np.float32)[keep]
        raw_timestamp = np.asarray(file["timestamp"], dtype=np.float64)
        source_frame_index = np.asarray(_require_dataset(file, "frame_index"), dtype=np.int64)[keep]
        timestamp_ns_ds = _first_existing_dataset(file, ("timestamp_ns",))
        if timestamp_ns_ds is None:
            source_timestamp_ns = np.full(keep.shape[0], -1, dtype=np.int64)
        else:
            source_timestamp_ns = np.asarray(timestamp_ns_ds, dtype=np.int64)[keep]
        policy_action = _read_array_feature(
            file,
            ("policy_action", "action.policy"),
            length=inspected.raw_length,
            shape=(16,),
            fill_value=np.nan,
        )[keep]
        teleop_action = _read_array_feature(
            file,
            ("teleop_action", "human_action", "action.human"),
            length=inspected.raw_length,
            shape=(16,),
            fill_value=np.nan,
        )[keep]

    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Filtered state/action contains NaN or Inf")

    length = inspected.length
    timestamp = _zeroed_timestamps(raw_timestamp, keep, fps=fps, mode=timestamp_mode)
    frame = pd.DataFrame(
        {
            ACTION_KEY: [row.copy() for row in action],
            STATE_KEY: [row.copy() for row in state],
            "complementary_info.is_intervention": inspected.human_mask_kept.astype(np.int64),
            "complementary_info.policy_action": [row.copy() for row in policy_action],
            "complementary_info.teleop_action": [row.copy() for row in teleop_action],
            "source_frame_index": source_frame_index.astype(np.int64),
            "source_timestamp_ns": source_timestamp_ns.astype(np.int64),
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
    return parquet_path, compute_episode_stats(frame)


def _video_shape_from_sources(
    sources: list[RawSource],
    inspected: list[InspectedEpisode],
    video_key: str,
) -> list[int]:
    for source in sources:
        feature = source.info.get("schema", {}).get("features", {}).get(video_key, {})
        shape = feature.get("shape")
        if shape:
            return [int(value) for value in shape]
    for item in inspected:
        if video_key in item.video_paths:
            return _read_video_shape(item.video_paths[video_key])
    raise ValueError(f"Cannot infer video shape for {video_key}")


def _feature_schema(
    sources: list[RawSource],
    inspected: list[InspectedEpisode],
    *,
    fps: int,
    video_codec: str,
) -> dict[str, Any]:
    def video_feature(video_key: str) -> dict[str, Any]:
        shape = _video_shape_from_sources(sources, inspected, video_key)
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
        "complementary_info.is_intervention": {"dtype": "int64", "shape": [1], "names": None},
        "complementary_info.policy_action": {"dtype": "float32", "shape": [16], "names": JOINT_NAMES},
        "complementary_info.teleop_action": {"dtype": "float32", "shape": [16], "names": JOINT_NAMES},
        "source_frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "source_timestamp_ns": {"dtype": "int64", "shape": [1], "names": None},
        "observation.images.left_wrist": video_feature("observation.images.left_wrist"),
        "observation.images.right_wrist": video_feature("observation.images.right_wrist"),
        "observation.images.base": video_feature("observation.images.base"),
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
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "total_chunks": max(1, math.ceil(total_episodes / chunks_size)),
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": splits,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _feature_schema(sources, inspected, fps=fps, video_codec=video_codec),
        "hil_conversion": {
            "dataset_id": dataset_id,
            "task": task,
            "mode": "evo_clean",
            "policy_keep_rule": "session_state=policy and authority_source=policy and selected_source=policy",
            "hold_drop_rule": "session_state=intervention_hold or selected_source=hold",
            "human_keep_rule": (
                "session_state=human and authority_source=human and selected_source=human and finite teleop_action"
            ),
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
    video_codec: str,
    overwrite: bool,
    dry_run: bool,
    episodes: str | None,
    max_episodes: int | None,
    min_frames: int,
    timestamp_mode: str,
    verify_video_frames: bool,
    skip_invalid: bool,
    default_success: str,
) -> dict[str, Any]:
    if default_success not in {"success", "failure"}:
        raise ValueError("--default-success must be success or failure")
    if val_count < 0:
        raise ValueError(f"--val-count must be >= 0, got {val_count}")

    sources = _discover_raw_sources(src_paths)
    candidates = _select_candidates(_collect_candidates(sources), episodes=episodes, max_episodes=max_episodes)
    inspected, rejected = _inspect_candidates(
        candidates,
        min_frames=min_frames,
        verify_video_frames=verify_video_frames,
        default_success=default_success,
    )
    if val_count >= len(inspected) and inspected:
        raise ValueError(f"--val-count {val_count} must be smaller than valid episode count {len(inspected)}")
    if rejected and not skip_invalid and not dry_run:
        _write_json(
            dst.with_name(f"{dst.name}_conversion_rejected_report.json"),
            {"destination": str(dst), "rejected": rejected},
        )
        raise ValueError(f"{len(rejected)} invalid episodes found; inspect rejection report or pass --skip-invalid")

    train_end = max(0, len(inspected) - val_count)
    report = {
        "destination": str(dst),
        "dataset_id": dataset_id,
        "task": task,
        "mode": "evo_clean",
        "dry_run": dry_run,
        "selected_episodes": len(candidates),
        "valid_episodes": len(inspected),
        "rejected_episodes": len(rejected),
        "raw_total_frames": sum(item.raw_length for item in inspected),
        "total_frames": sum(item.length for item in inspected),
        "policy_frames": sum(item.policy_count for item in inspected),
        "human_frames": sum(item.human_count for item in inspected),
        "dropped_hold_frames": sum(item.hold_count for item in inspected),
        "dropped_other_frames": sum(item.dropped_other_count for item in inspected),
        "fps": fps,
        "timestamp_mode": timestamp_mode,
        "video_codec": video_codec,
        "splits": {"train": f"0:{train_end}", **({"val": f"{train_end}:{len(inspected)}"} if val_count else {})},
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
    episode_stats_rows: list[dict[str, Any]] = []
    for new_episode_index, item in enumerate(inspected):
        _, stats = _write_episode_parquet(
            item,
            dst,
            episode_index=new_episode_index,
            global_start_index=total_frames,
            chunks_size=chunks_size,
            fps=fps,
            timestamp_mode=timestamp_mode,
        )
        for video_key, src_video in item.video_paths.items():
            _write_filtered_video(
                src_video,
                dst / _format_video_path(new_episode_index, chunks_size, video_key),
                keep_indices=item.keep_indices,
                fps=fps,
                codec=video_codec,
            )

        raw_meta = dict(item.candidate.raw_meta)
        episode_row = {
            "episode_index": new_episode_index,
            "tasks": [task],
            "length": item.length,
            "source_dataset": item.candidate.source.name,
            "source_episode_index": item.candidate.source_episode_index,
            "source_hdf5_path": str(item.candidate.hdf5_path),
            "raw_length": item.raw_length,
            "policy_frames": item.policy_count,
            "human_frames": item.human_count,
            "dropped_hold_frames": item.hold_count,
            "dropped_other_frames": item.dropped_other_count,
            "episode_success": _episode_success_from_meta(raw_meta, default_success),
            "episode_outcome": raw_meta.get("episode_outcome"),
            "recovery_success": raw_meta.get("recovery_success"),
            "collector_policy_id": raw_meta.get("collector_policy_id"),
            "model_metadata": raw_meta.get("model_metadata"),
            "timestamp_start": item.timestamp_start,
            "timestamp_end": item.timestamp_end,
            "source_frame_start": int(item.keep_indices[0]),
            "source_frame_end": int(item.keep_indices[-1]),
        }
        for optional_key in ("intervention_count", "intervention_start_frames", "intervention_end_frames"):
            if optional_key in raw_meta:
                episode_row[optional_key] = raw_meta[optional_key]
        episode_rows.append(episode_row)
        episode_stats_rows.append({"episode_index": new_episode_index, "stats": stats})
        total_frames += item.length

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
    _write_jsonl(dst / "meta/episodes_stats.jsonl", episode_stats_rows)
    _write_json(dst / "conversion_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--dst", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--episodes", default=None, help="Flattened candidate spec, e.g. 0:100 or 0,2,5.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--video-codec", default="mp4v", help="FourCC used by OpenCV when rewriting filtered videos.")
    parser.add_argument("--timestamp-mode", choices=("raw_zeroed", "fps"), default="fps")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--verify-video-frames", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--default-success", choices=("success", "failure"), default="failure")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    src_paths = args.src or [pathlib.Path("/tmp/openarm_hil/openarm_hil_dagger")]
    convert_dataset(
        src_paths,
        args.dst,
        dataset_id=args.dataset_id,
        task=args.task,
        val_count=args.val_count,
        fps=args.fps,
        chunks_size=args.chunks_size,
        video_codec=args.video_codec,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        episodes=args.episodes,
        max_episodes=args.max_episodes,
        min_frames=args.min_frames,
        timestamp_mode=args.timestamp_mode,
        verify_video_frames=args.verify_video_frames,
        skip_invalid=args.skip_invalid,
        default_success=args.default_success,
    )


if __name__ == "__main__":
    main()
