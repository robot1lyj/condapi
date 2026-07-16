"""Audit formal OpenArm KAI0 AWBC data with real OpenPI loader and video samples."""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

TASKS = (
    {"task_index": 0, "task": "Fold the T-shirt properly, Advantage: negative"},
    {"task_index": 1, "task": "Fold the T-shirt properly, Advantage: positive"},
)
DEFAULT_SOURCE_COUNTS = {"HQ": 999, "Site": 420, "TDA": 300}


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _data_path(info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunk = episode_index // int(info["chunks_size"])
    return pathlib.Path(info["data_path"].format(episode_chunk=chunk, episode_index=episode_index))


def audit_dataset_structure(
    dataset: pathlib.Path,
    *,
    expected_episodes: int,
    expected_source_counts: dict[str, int],
) -> dict[str, Any]:
    dataset = dataset.resolve()
    info = _load_json(dataset / "meta/info.json")
    tasks = _load_jsonl(dataset / "meta/tasks.jsonl")
    episodes = _load_jsonl(dataset / "meta/episodes.jsonl")
    episode_stats = _load_jsonl(dataset / "meta/episodes_stats.jsonl")
    build_report = _load_json(dataset / "kai0_awbc_build_report.json")

    if tasks != list(TASKS):
        raise ValueError(f"K-Data tasks must be the formal binary prompts, got {tasks}")
    if int(info.get("total_episodes", -1)) != expected_episodes or len(episodes) != expected_episodes:
        raise ValueError(
            f"K-Data episode count mismatch: info={info.get('total_episodes')}, "
            f"metadata={len(episodes)}, expected={expected_episodes}"
        )
    episode_ids = [int(row["episode_index"]) for row in episodes]
    if episode_ids != list(range(expected_episodes)):
        raise ValueError("K-Data episode indices must be contiguous from zero")
    stats_episode_ids = [int(row["episode_index"]) for row in episode_stats]
    if stats_episode_ids != episode_ids:
        raise ValueError("K-Data episode stats must cover every contiguous episode")

    source_counts = Counter(str(row.get("source_kind")) for row in episodes)
    if dict(source_counts) != expected_source_counts:
        raise ValueError(f"K-Data source counts mismatch: {dict(source_counts)} != {expected_source_counts}")
    if int(build_report.get("total_episodes", -1)) != expected_episodes:
        raise ValueError("K-Data build report does not match the expected episode count")
    if build_report.get("tasks") != list(TASKS):
        raise ValueError("K-Data build report does not contain the formal binary prompts")
    if build_report.get("advantage_source") != "absolute_advantage":
        raise ValueError("K-Data must discretize the official KAI0 absolute_advantage source")

    label_counts: Counter[int] = Counter()
    selected_indices: dict[int, int] = {}
    episodes_by_source: dict[str, list[int]] = {}
    for row in episodes:
        episodes_by_source.setdefault(str(row["source_kind"]), []).append(int(row["episode_index"]))
    source_sample_episodes = {
        source: {episode_ids[0], episode_ids[len(episode_ids) // 2], episode_ids[-1]}
        for source, episode_ids in episodes_by_source.items()
    }
    selected_source_indices: dict[str, list[int]] = {source: [] for source in episodes_by_source}
    tda_tail_indices: list[int] = []
    expected_global_index = 0
    total_frames = 0
    for episode_index in range(expected_episodes):
        frame = pd.read_parquet(
            dataset / _data_path(info, episode_index),
            columns=["index", "episode_index", "frame_index", "task_index"],
        )
        length = len(frame)
        if length <= 0:
            raise ValueError(f"K-Data episode {episode_index} is empty")
        actual_episode_ids = frame["episode_index"].to_numpy(dtype=np.int64)
        if not np.all(actual_episode_ids == episode_index):
            raise ValueError(f"K-Data episode {episode_index} contains mismatched episode_index values")
        actual_frame_ids = frame["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_frame_ids, np.arange(length, dtype=np.int64)):
            raise ValueError(f"K-Data episode {episode_index} frame_index is not contiguous")
        actual_global_ids = frame["index"].to_numpy(dtype=np.int64)
        expected_ids = np.arange(expected_global_index, expected_global_index + length, dtype=np.int64)
        if not np.array_equal(actual_global_ids, expected_ids):
            raise ValueError(f"K-Data episode {episode_index} global index is not contiguous")

        labels = frame["task_index"].to_numpy(dtype=np.int64)
        unexpected = set(np.unique(labels).tolist()) - {0, 1}
        if unexpected:
            raise ValueError(f"K-Data episode {episode_index} has non-binary labels: {sorted(unexpected)}")
        label_counts.update(labels.tolist())
        for label in (0, 1):
            if label not in selected_indices:
                positions = np.flatnonzero(labels == label)
                if len(positions):
                    selected_indices[label] = expected_global_index + int(positions[0])
        source = str(episodes[episode_index]["source_kind"])
        if source == "TDA":
            tda_tail_indices.append(expected_global_index + length - 1)
        if episode_index in source_sample_episodes[source]:
            for frame_offset in (length // 2, length - 1):
                global_index = expected_global_index + frame_offset
                if global_index not in selected_source_indices[source]:
                    selected_source_indices[source].append(global_index)
        expected_global_index += length
        total_frames += length

    if set(label_counts) != {0, 1}:
        raise ValueError(f"K-Data must contain both binary labels, got {dict(label_counts)}")
    if total_frames != int(info.get("total_frames", -1)):
        raise ValueError(f"K-Data frame count mismatch: scanned={total_frames}, info={info.get('total_frames')}")

    return {
        "dataset": str(dataset),
        "total_episodes": expected_episodes,
        "total_frames": total_frames,
        "source_counts": dict(source_counts),
        "label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "positive_ratio": label_counts[1] / total_frames,
        "selected_global_indices": {str(key): value for key, value in selected_indices.items()},
        "selected_source_global_indices": selected_source_indices,
        "tda_tail_global_indices": tda_tail_indices,
    }


def run_openpi_loader_smoke(
    dataset: pathlib.Path,
    *,
    config_name: str,
    expected_episodes: int,
    selected_global_indices: dict[str, int],
    selected_source_global_indices: dict[str, list[int]],
    tda_tail_global_indices: list[int],
) -> dict[str, Any]:
    train_config = _config.get_config(config_name)
    base_data_config = train_config.data.base_config or _config.DataConfig()
    data_factory = dataclasses.replace(
        train_config.data,
        repo_id=str(dataset.resolve()),
        assets=_config.AssetsConfig(assets_dir=str(dataset.resolve().parent), asset_id=dataset.name),
        base_config=dataclasses.replace(base_data_config, train_episodes=list(range(expected_episodes))),
    )
    train_config = dataclasses.replace(train_config, data=data_factory, batch_size=2, num_workers=0)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise ValueError(f"K-Data norm stats were not loaded from {dataset / 'norm_stats.json'}")
    if data_config.lerobot_video_backend != "pyav" or data_config.lerobot_tolerance_s != 0.05:
        raise ValueError(
            "K-Data must use the validated PyAV/0.05s mixed-source video contract, got "
            f"backend={data_config.lerobot_video_backend!r}, tolerance={data_config.lerobot_tolerance_s!r}"
        )

    data_transform_names = [type(transform).__name__ for transform in data_config.data_transforms.inputs]
    if "OpenArmInputs" not in data_transform_names or any("Piper" in name for name in data_transform_names):
        raise ValueError(f"K-Data must use only the OpenArm input path, got {data_transform_names}")

    raw_dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config)
    samples: dict[str, Any] = {}
    token_sequences = {}
    for label, expected_prompt in ((0, TASKS[0]["task"]), (1, TASKS[1]["task"])):
        index = int(selected_global_indices[str(label)])
        raw = raw_dataset[index]
        prompt = raw["prompt"].item() if not isinstance(raw["prompt"], str) else raw["prompt"]
        if prompt != expected_prompt:
            raise ValueError(f"Label {label} resolved to prompt {prompt!r}, expected {expected_prompt!r}")
        raw_state = np.asarray(raw["observation.state"])
        raw_actions = np.asarray(raw["action"])
        if raw_state.shape != (16,) or raw_actions.shape != (train_config.model.action_horizon, 16):
            raise ValueError(
                f"Label {label} raw OpenArm shapes are invalid: state={raw_state.shape}, actions={raw_actions.shape}"
            )
        if not np.isfinite(raw_state).all() or not np.isfinite(raw_actions).all():
            raise ValueError(f"Label {label} contains non-finite state/action values")

        transformed = transformed_dataset[index]
        transformed_state = np.asarray(transformed["state"])
        transformed_actions = np.asarray(transformed["actions"])
        if transformed_state.shape != (train_config.model.action_dim,):
            raise ValueError(f"Label {label} transformed state shape is {transformed_state.shape}")
        expected_action_shape = (train_config.model.action_horizon, train_config.model.action_dim)
        if transformed_actions.shape != expected_action_shape:
            raise ValueError(f"Label {label} transformed action shape is {transformed_actions.shape}")
        tokens = np.asarray(transformed["tokenized_prompt"])
        mask = np.asarray(transformed["tokenized_prompt_mask"], dtype=bool)
        token_sequences[label] = tokens[mask]
        samples[str(label)] = {
            "global_index": index,
            "prompt": prompt,
            "raw_state_shape": list(raw_state.shape),
            "raw_action_shape": list(raw_actions.shape),
            "model_state_shape": list(transformed_state.shape),
            "model_action_shape": list(transformed_actions.shape),
            "prompt_token_count": int(mask.sum()),
        }

    if np.array_equal(token_sequences[0], token_sequences[1]):
        raise ValueError("Positive and negative AWBC prompts produced identical token sequences")

    source_video_samples = {}
    for source, indices in selected_source_global_indices.items():
        source_video_samples[source] = []
        for index in indices:
            transformed = transformed_dataset[int(index)]
            images = transformed["image"]
            image_shapes = {key: list(np.asarray(value).shape) for key, value in images.items()}
            if not all(np.isfinite(np.asarray(value)).all() for value in images.values()):
                raise ValueError(f"{source} video sample {index} contains non-finite pixels")
            source_video_samples[source].append({"global_index": int(index), "image_shapes": image_shapes})

    # TDA videos are regenerated independently from their parquet metadata. Decode every
    # episode tail because an MP4 header can advertise one more frame than it contains.
    for index in tda_tail_global_indices:
        transformed = transformed_dataset[int(index)]
        images = transformed["image"]
        if not all(np.isfinite(np.asarray(value)).all() for value in images.values()):
            raise ValueError(f"TDA tail video sample {index} contains non-finite pixels")

    return {
        "config": config_name,
        "video_backend": data_config.lerobot_video_backend,
        "video_tolerance_s": data_config.lerobot_tolerance_s,
        "data_transforms": data_transform_names,
        "model_transforms": [type(transform).__name__ for transform in data_config.model_transforms.inputs],
        "samples": samples,
        "source_video_samples": source_video_samples,
        "tda_tail_samples": {
            "count": len(tda_tail_global_indices),
            "first_global_index": int(tda_tail_global_indices[0]) if tda_tail_global_indices else None,
            "last_global_index": int(tda_tail_global_indices[-1]) if tda_tail_global_indices else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--config", default="pi05_openarm_kai0_awbc_v1")
    parser.add_argument("--expected-episodes", type=int, default=1719)
    parser.add_argument("--expected-hq", type=int, default=999)
    parser.add_argument("--expected-site", type=int, default=420)
    parser.add_argument("--expected-tda", type=int, default=300)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    structure = audit_dataset_structure(
        args.dataset,
        expected_episodes=args.expected_episodes,
        expected_source_counts={"HQ": args.expected_hq, "Site": args.expected_site, "TDA": args.expected_tda},
    )
    loader = run_openpi_loader_smoke(
        args.dataset,
        config_name=args.config,
        expected_episodes=args.expected_episodes,
        selected_global_indices=structure["selected_global_indices"],
        selected_source_global_indices=structure["selected_source_global_indices"],
        tda_tail_global_indices=structure["tda_tail_global_indices"],
    )
    report = {"passed": True, "structure": structure, "loader": loader}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
