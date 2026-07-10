"""Build an OpenArm AWBC dataset with Stage Advantage predictions.

This is the OpenArm equivalent of KAI0 Stage Advantage Step 2 + Step 3:

1. Use a trained Stage Advantage estimator to append advantage columns:
   relative_advantage, absolute_value, absolute_advantage.
2. In legacy one-shot mode, discretize the scores into task labels.
   ``--score-only`` skips this step so parallel shards can be merged before
   the official binary threshold is computed once over the full dataset.
3. Write meta/tasks.jsonl prompts consumed by prompt_from_task=True.

The source dataset is never modified. Videos are copied/linked into the output
dataset, while parquet files are rewritten with contiguous episode indices.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import dataclasses
import json
import logging
import math
import os
import pathlib
import shutil
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pandas as pd
import tqdm

if TYPE_CHECKING:
    import torch

    import openpi.training.config as _config

OPENARM_VIDEO_KEYS = (
    "observation.images.base",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
MODEL_IMAGE_KEYS = (
    ("base", "base_-100_rgb", "base_0_rgb"),
    ("left_wrist", "left_wrist_-100_rgb", "left_wrist_0_rgb"),
    ("right_wrist", "right_wrist_-100_rgb", "right_wrist_0_rgb"),
)
AWBC_TASKS = (
    (0, "bad"),
    (1, "neutral"),
    (2, "positive"),
)
SCORE_COLUMNS = ("relative_advantage", "absolute_value", "absolute_advantage")
VIDEO_IO_ATTEMPTS = 3
VIDEO_IO_RETRY_DELAY_SECONDS = 2.0


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def parse_episodes(spec: str | None, total_episodes: int) -> list[int]:
    if spec is None:
        return list(range(total_episodes))
    spec = spec.strip()
    if not spec:
        raise ValueError("Episode spec is empty")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid range episode spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, end, step))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_data_path(info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(info["data_path"].format(episode_chunk=chunk, episode_index=episode_index))


def _format_video_path(info: dict[str, Any], episode_index: int, video_key: str) -> pathlib.Path:
    chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return pathlib.Path(
        info["video_path"].format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)
    )


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


def _write_parquet_atomic(frame: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scored_episode_is_complete(
    path: pathlib.Path,
    *,
    expected_length: int,
    expected_episode_index: int,
    expected_start_index: int,
) -> bool:
    if not path.exists():
        return False
    required_columns = (*SCORE_COLUMNS, "episode_index", "frame_index", "index")
    try:
        frame = pd.read_parquet(path, columns=list(required_columns))
    except Exception:
        return False
    if len(frame) != expected_length:
        return False
    if expected_length == 0:
        return False
    if not all(np.isfinite(frame[column].to_numpy(dtype=np.float32)).all() for column in SCORE_COLUMNS):
        return False
    return bool(
        np.all(frame["episode_index"].to_numpy(dtype=np.int64) == expected_episode_index)
        and np.array_equal(frame["frame_index"].to_numpy(dtype=np.int64), np.arange(expected_length))
        and np.array_equal(
            frame["index"].to_numpy(dtype=np.int64),
            np.arange(expected_start_index, expected_start_index + expected_length),
        )
    )


class VideoFrameReader:
    def __init__(
        self,
        video_path: pathlib.Path,
        *,
        io_attempts: int = VIDEO_IO_ATTEMPTS,
        retry_delay_seconds: float = VIDEO_IO_RETRY_DELAY_SECONDS,
    ):
        if io_attempts < 1:
            raise ValueError("Video I/O attempts must be positive")
        self._cv2 = cv2
        self._video_path = video_path
        self._io_attempts = io_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._cap = self._open_capture()
        self._next_index = 0
        self._cache: dict[int, np.ndarray] = {}

    def _open_capture(self):
        for attempt in range(1, self._io_attempts + 1):
            capture = self._cv2.VideoCapture(str(self._video_path))
            if capture.isOpened():
                return capture
            capture.release()
            if attempt < self._io_attempts:
                logging.warning(
                    "Could not open video %s (attempt %d/%d); retrying",
                    self._video_path,
                    attempt,
                    self._io_attempts,
                )
                time.sleep(self._retry_delay_seconds)
        raise RuntimeError(f"Could not open video after {self._io_attempts} attempts: {self._video_path}")

    def _reopen(self) -> None:
        self._cap.release()
        time.sleep(self._retry_delay_seconds)
        self._cap = self._open_capture()
        self._next_index = 0

    def read(self, frame_index: int) -> np.ndarray:
        if frame_index in self._cache:
            return self._cache[frame_index]
        frame = None
        for attempt in range(1, self._io_attempts + 1):
            if frame_index != self._next_index:
                self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_index)
                self._next_index = frame_index
            ok, frame = self._cap.read()
            if ok:
                break
            if attempt < self._io_attempts:
                logging.warning(
                    "Could not read frame %d from %s (attempt %d/%d); reopening",
                    frame_index,
                    self._video_path,
                    attempt,
                    self._io_attempts,
                )
                self._reopen()
        else:
            raise RuntimeError(
                f"Could not read frame {frame_index} after {self._io_attempts} attempts from {self._video_path}"
            )
        self._next_index = frame_index + 1
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        if len(self._cache) > 128:
            self._cache.pop(next(iter(self._cache)))
        self._cache[frame_index] = rgb
        return rgb

    def close(self) -> None:
        self._cap.release()


def _image_batch(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    import torch  # noqa: PLC0415

    from openpi.shared import image_tools  # noqa: PLC0415

    tensor = torch.from_numpy(np.stack(frames)).to(device=device, dtype=torch.float32) / 255.0
    tensor = tensor * 2.0 - 1.0
    tensor = image_tools.resize_with_pad_torch(tensor, 224, 224)
    return tensor.permute(0, 3, 1, 2)


def _stack_column(values: pd.Series, *, expected_dim: int, key: str) -> np.ndarray:
    array = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values.to_numpy()])
    if array.shape[-1] != expected_dim:
        raise ValueError(f"{key} expected {expected_dim}D, got {array.shape[-1]}D")
    return array


def _pad_state(states: np.ndarray, action_dim: int) -> np.ndarray:
    if states.shape[-1] > action_dim:
        raise ValueError(f"State dim {states.shape[-1]} exceeds model action_dim {action_dim}")
    if states.shape[-1] == action_dim:
        return states.astype(np.float32)
    padded = np.zeros((states.shape[0], action_dim), dtype=np.float32)
    padded[:, : states.shape[-1]] = states.astype(np.float32)
    return padded


def _build_observation(
    *,
    readers: dict[str, VideoFrameReader],
    history_indices: list[int],
    current_indices: list[int],
    states: np.ndarray,
    tokenized_prompt: np.ndarray,
    tokenized_prompt_mask: np.ndarray,
    action_dim: int,
    device: torch.device,
) -> SimpleNamespace:
    import torch  # noqa: PLC0415

    images: dict[str, torch.Tensor] = {}
    for camera_name, history_key, current_key in MODEL_IMAGE_KEYS:
        reader = readers[camera_name]
        history_frames = [reader.read(idx) for idx in history_indices]
        current_frames = [reader.read(idx) for idx in current_indices]
        images[history_key] = _image_batch(history_frames, device)
        images[current_key] = _image_batch(current_frames, device)

    batch_size = len(current_indices)
    state_batch = torch.from_numpy(_pad_state(states, action_dim)).to(device=device, dtype=torch.float32)
    tokens_batch = np.tile(tokenized_prompt[np.newaxis, :], (batch_size, 1))
    token_masks_batch = np.tile(tokenized_prompt_mask[np.newaxis, :], (batch_size, 1))

    return SimpleNamespace(
        images=images,
        image_masks={key: torch.ones((batch_size,), dtype=torch.bool, device=device) for key in images},
        state=state_batch,
        tokenized_prompt=torch.from_numpy(tokens_batch).to(device),
        tokenized_prompt_mask=torch.from_numpy(token_masks_batch).to(device),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def _load_model(
    config_name: str, checkpoint: pathlib.Path, device: torch.device
) -> tuple[_config.TrainConfig, torch.nn.Module]:
    import safetensors.torch  # noqa: PLC0415

    import openpi.models.pi0_config as pi0_config  # noqa: PLC0415
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch  # noqa: PLC0415
    import openpi.training.config as _config  # noqa: PLC0415

    config = _config.get_config(config_name)
    model_config = dataclasses.replace(config.model, dtype=config.pytorch_training_precision)
    if not isinstance(model_config, pi0_config.AdvantageEstimatorConfig):
        raise TypeError(f"{config_name} must use AdvantageEstimatorConfig")
    config = dataclasses.replace(config, model=model_config)
    model = pi0_pytorch.AdvantageEstimator(model_config).to(device)
    model_path = checkpoint / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint model: {model_path}")
    safetensors.torch.load_model(model, model_path, strict=True)
    model.eval()
    return config, model


def _predict_episode(
    *,
    src: pathlib.Path,
    info: dict[str, Any],
    episode_index: int,
    df: pd.DataFrame,
    config: _config.TrainConfig,
    model: torch.nn.Module,
    device: torch.device,
    prompt: str,
    batch_size: int,
    relative_interval: int,
    samples_per_batch: int,
    seed: int,
) -> dict[str, np.ndarray]:
    import torch  # noqa: PLC0415

    import openpi.models.tokenizer as _tokenizer  # noqa: PLC0415

    frame_count = len(df)
    states = _stack_column(df["observation.state"], expected_dim=16, key="observation.state")
    tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)
    tokenized_prompt, tokenized_prompt_mask = tokenizer.tokenize(prompt, state=None)
    readers = {}
    for camera_name, _, _ in MODEL_IMAGE_KEYS:
        readers[camera_name] = VideoFrameReader(
            src / _format_video_path(info, episode_index, f"observation.images.{camera_name}")
        )

    absolute_value = np.zeros((frame_count,), dtype=np.float32)
    relative_advantage = np.zeros((frame_count,), dtype=np.float32)

    try:
        with torch.inference_mode():
            for batch_start in tqdm.tqdm(
                range(0, frame_count, batch_size), desc=f"episode {episode_index}", leave=False
            ):
                batch_end = min(batch_start + batch_size, frame_count)
                current_indices = list(range(batch_start, batch_end))
                future_indices = [min(idx + relative_interval, frame_count - 1) for idx in current_indices]

                relative_obs = _build_observation(
                    readers=readers,
                    history_indices=current_indices,
                    current_indices=future_indices,
                    states=states[future_indices],
                    tokenized_prompt=tokenized_prompt,
                    tokenized_prompt_mask=tokenized_prompt_mask,
                    action_dim=config.model.action_dim,
                    device=device,
                )
                absolute_obs = _build_observation(
                    readers=readers,
                    history_indices=[0] * len(current_indices),
                    current_indices=current_indices,
                    states=states[current_indices],
                    tokenized_prompt=tokenized_prompt,
                    tokenized_prompt_mask=tokenized_prompt_mask,
                    action_dim=config.model.action_dim,
                    device=device,
                )

                relative_accum = torch.zeros((len(current_indices),), dtype=torch.float32, device=device)
                absolute_accum = torch.zeros((len(current_indices),), dtype=torch.float32, device=device)
                for sample_idx in range(samples_per_batch):
                    torch.manual_seed(seed + episode_index * 100_000 + batch_start * samples_per_batch + sample_idx)
                    relative_accum += model.sample_values(device, relative_obs).to(torch.float32).reshape(-1)
                    absolute_accum += model.sample_values(device, absolute_obs).to(torch.float32).reshape(-1)

                relative_vals = (relative_accum / samples_per_batch).detach().cpu().numpy()
                absolute_vals = (absolute_accum / samples_per_batch).detach().cpu().numpy()
                relative_advantage[current_indices] = relative_vals
                absolute_value[current_indices] = absolute_vals
    finally:
        for reader in readers.values():
            reader.close()

    absolute_value[0] = 0.0
    absolute_advantage = np.zeros((frame_count,), dtype=np.float32)
    for frame_idx in range(frame_count):
        future_idx = min(frame_idx + relative_interval, frame_count - 1)
        if future_idx == frame_idx:
            absolute_advantage[frame_idx] = 0.0
            relative_advantage[frame_idx] = 0.0
            continue
        delta = future_idx - frame_idx
        scale = relative_interval / delta if delta != relative_interval else 1.0
        absolute_advantage[frame_idx] = (absolute_value[future_idx] - absolute_value[frame_idx]) * scale
        if delta != relative_interval:
            relative_advantage[frame_idx] *= scale

    return {
        "relative_advantage": np.clip(relative_advantage, -1.0, 1.0).astype(np.float32),
        "absolute_value": np.clip(absolute_value, -1.0, 1.0).astype(np.float32),
        "absolute_advantage": np.clip(absolute_advantage, -1.0, 1.0).astype(np.float32),
    }


def _add_awbc_features(info: dict[str, Any]) -> dict[str, Any]:
    new_info = dict(info)
    features = dict(new_info["features"])
    for key in ("relative_advantage", "absolute_value", "absolute_advantage"):
        features[key] = {"dtype": "float32", "shape": [1], "names": None}
    new_info["features"] = features
    return new_info


def _write_tasks(dst: pathlib.Path, task: str) -> None:
    rows = [{"task_index": task_index, "task": f"{task}, Advantage: {name}"} for task_index, name in AWBC_TASKS]
    _write_jsonl(dst / "meta/tasks.jsonl", rows)


def _write_base_task(dst: pathlib.Path, task: str) -> None:
    _write_jsonl(dst / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])


def _assign_awbc_labels(
    dst: pathlib.Path, parquets: list[pathlib.Path], *, bad_percentile: float, positive_percentile: float
) -> dict:
    values = []
    for parquet_path in parquets:
        episode_frame = pd.read_parquet(parquet_path, columns=["absolute_advantage"])
        values.append(episode_frame["absolute_advantage"].to_numpy(dtype=np.float32))
    all_values = np.concatenate(values)
    bad_threshold = float(np.percentile(all_values, bad_percentile))
    positive_threshold = float(np.percentile(all_values, positive_percentile))
    if bad_threshold > positive_threshold:
        raise ValueError("--bad-percentile must be <= --positive-percentile")

    counts = {name: 0 for _, name in AWBC_TASKS}
    for parquet_path in parquets:
        episode_frame = pd.read_parquet(parquet_path)
        advantage = episode_frame["absolute_advantage"].to_numpy(dtype=np.float32)
        task_index = np.ones((len(episode_frame),), dtype=np.int64)
        task_index[advantage <= bad_threshold] = 0
        task_index[advantage >= positive_threshold] = 2
        episode_frame["task_index"] = task_index
        episode_frame.to_parquet(parquet_path, index=False)
        for idx, name in AWBC_TASKS:
            counts[name] += int(np.sum(task_index == idx))

    report = {
        "bad_percentile": bad_percentile,
        "positive_percentile": positive_percentile,
        "bad_threshold": bad_threshold,
        "positive_threshold": positive_threshold,
        "counts": counts,
        "total_frames": len(all_values),
    }
    _write_json(dst / "awbc_discretize_report.json", report)
    return report


def build_awbc_dataset(args: argparse.Namespace) -> dict[str, Any]:
    src = args.src.resolve()
    dst = args.dst.resolve()
    info = _load_json(src / "meta/info.json")
    episodes = parse_episodes(args.episodes, int(info["total_episodes"]))

    for video_key in OPENARM_VIDEO_KEYS:
        if video_key not in info["features"]:
            raise ValueError(f"Source dataset missing video key: {video_key}")

    resume = bool(getattr(args, "resume", False))
    if args.overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    if args.dry_run:
        report = {
            "source": str(src),
            "destination": str(dst),
            "episodes": episodes,
            "episode_count": len(episodes),
            "checkpoint": str(args.checkpoint),
            "config_name": args.config_name,
            "task": args.task,
            "score_only": args.score_only,
            "resume": resume,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    if dst.exists() and args.overwrite:
        shutil.rmtree(dst)
    elif dst.exists() and not resume:
        raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it or --resume to continue it")
    dst.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: PLC0415

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    config, model = _load_model(args.config_name, args.checkpoint.resolve(), device)
    episodes_meta = {int(row["episode_index"]): row for row in _load_jsonl(src / "meta/episodes.jsonl")}
    stats_path = src / "meta/episodes_stats.jsonl"
    stats_meta = {}
    if stats_path.exists():
        stats_meta = {int(row["episode_index"]): row for row in _load_jsonl(stats_path)}

    new_episode_rows: list[dict[str, Any]] = []
    new_stats_rows: list[dict[str, Any]] = []
    output_parquets: list[pathlib.Path] = []
    total_frames = 0
    total_videos = 0
    resumed_episodes = 0
    newly_scored_episodes = 0

    for new_episode_index, old_episode_index in enumerate(episodes):
        old_parquet = src / _format_data_path(info, old_episode_index)
        new_parquet = dst / _format_data_path(info, new_episode_index)
        episode_frame = pd.read_parquet(old_parquet).copy()
        length = len(episode_frame)
        already_complete = resume and _scored_episode_is_complete(
            new_parquet,
            expected_length=length,
            expected_episode_index=new_episode_index,
            expected_start_index=total_frames,
        )
        if already_complete:
            resumed_episodes += 1
        else:
            predictions = _predict_episode(
                src=src,
                info=info,
                episode_index=old_episode_index,
                df=episode_frame,
                config=config,
                model=model,
                device=device,
                prompt=args.task,
                batch_size=args.batch_size,
                relative_interval=args.relative_interval,
                samples_per_batch=args.samples_per_batch,
                seed=args.seed,
            )
            episode_frame["episode_index"] = np.full(length, new_episode_index, dtype=np.int64)
            episode_frame["frame_index"] = np.arange(length, dtype=np.int64)
            episode_frame["index"] = np.arange(total_frames, total_frames + length, dtype=np.int64)
            for key, value in predictions.items():
                episode_frame[key] = value
            _write_parquet_atomic(episode_frame, new_parquet)
            newly_scored_episodes += 1
        output_parquets.append(new_parquet)

        episode_row = dict(episodes_meta[old_episode_index])
        episode_row["episode_index"] = new_episode_index
        episode_row["tasks"] = (
            [args.task]
            if args.score_only
            else [
                f"{args.task}, Advantage: bad",
                f"{args.task}, Advantage: neutral",
                f"{args.task}, Advantage: positive",
            ]
        )
        episode_row["length"] = length
        episode_row["source_episode_index"] = old_episode_index
        new_episode_rows.append(episode_row)

        if old_episode_index in stats_meta:
            stats_row = dict(stats_meta[old_episode_index])
            stats_row["episode_index"] = new_episode_index
            new_stats_rows.append(stats_row)

        for video_key in OPENARM_VIDEO_KEYS:
            old_video = src / _format_video_path(info, old_episode_index, video_key)
            new_video = dst / _format_video_path(info, new_episode_index, video_key)
            _copy_or_link(old_video, new_video, args.copy_mode)
            total_videos += 1

        total_frames += length

    discretize_report = None
    if not args.score_only:
        discretize_report = _assign_awbc_labels(
            dst,
            output_parquets,
            bad_percentile=args.bad_percentile,
            positive_percentile=args.positive_percentile,
        )

    new_info = _add_awbc_features(info)
    new_info["total_episodes"] = len(episodes)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = total_videos
    new_info["total_chunks"] = max(1, math.ceil(len(episodes) / int(info["chunks_size"])))
    new_info["splits"] = {"train": f"0:{len(episodes)}"}
    new_info["total_tasks"] = 1 if args.score_only else len(AWBC_TASKS)
    _write_json(dst / "meta/info.json", new_info)
    if args.score_only:
        _write_base_task(dst, args.task)
    else:
        _write_tasks(dst, args.task)
    _write_jsonl(dst / "meta/episodes.jsonl", new_episode_rows)
    if new_stats_rows:
        _write_jsonl(dst / "meta/episodes_stats.jsonl", new_stats_rows)

    report = {
        "source": str(src),
        "destination": str(dst),
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "episodes": episodes,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_videos": total_videos,
        "relative_interval": args.relative_interval,
        "samples_per_batch": args.samples_per_batch,
        "score_only": args.score_only,
        "resume": resume,
        "resumed_episodes": resumed_episodes,
        "newly_scored_episodes": newly_scored_episodes,
        "discretize": discretize_report,
    }
    _write_json(dst / "awbc_build_report.json", report)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=pathlib.Path, required=True, help="Source LeRobot dataset")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="Destination AWBC LeRobot dataset")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Stage Advantage checkpoint directory")
    parser.add_argument("--config-name", default="ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD")
    parser.add_argument("--episodes", default=None, help="Episode spec, e.g. 0:10 or 0,2,5")
    parser.add_argument("--task", default="Fold the T-shirt properly")
    parser.add_argument("--relative-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-batch", type=int, default=1)
    parser.add_argument("--bad-percentile", type=float, default=20.0)
    parser.add_argument("--positive-percentile", type=float, default=70.0)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Write raw Stage predictions without assigning task labels; intended for parallel shards.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep valid completed episodes and continue the rest")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = build_awbc_dataset(args)
    if not args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
