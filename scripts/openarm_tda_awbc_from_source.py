"""Build an AWBC dataset for TDA-augmented OpenArm data by source mapping.

This avoids asking the Stage Advantage model to re-score mirrored/time-scaled
videos. The model predicts values only on original source episodes, then this
script maps those values to augmented episodes through explicit
`source_episode_index` / `augmentation_type` metadata.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
import pathlib
import shutil
from typing import Any

import numpy as np
from openarm_stage_advantage_awbc import OPENARM_VIDEO_KEYS
from openarm_stage_advantage_awbc import _add_awbc_features
from openarm_stage_advantage_awbc import _assign_awbc_labels
from openarm_stage_advantage_awbc import _copy_or_link
from openarm_stage_advantage_awbc import _format_data_path
from openarm_stage_advantage_awbc import _format_video_path
from openarm_stage_advantage_awbc import _load_json
from openarm_stage_advantage_awbc import _load_jsonl
from openarm_stage_advantage_awbc import _load_model
from openarm_stage_advantage_awbc import _predict_episode
from openarm_stage_advantage_awbc import _write_json
from openarm_stage_advantage_awbc import _write_jsonl
from openarm_stage_advantage_awbc import _write_tasks
from openarm_stage_advantage_awbc import parse_episodes
import pandas as pd
import tqdm


def _infer_mapping_from_order(
    row: dict[str, Any], report: dict[str, Any], default_extraction_factor: int
) -> dict[str, Any]:
    episode_index = int(row["episode_index"])
    source = report["source"]
    source_episodes = int(report["source_episodes"])
    time_scaled_episodes = int(report["time_scaled_episodes"])
    extraction_factor = int(report.get("time_extraction_factor", default_extraction_factor))

    if episode_index < source_episodes:
        return {
            "source_dataset": source,
            "source_episode_index": episode_index,
            "augmentation_type": "original",
            "source_frame_stride": 1,
            "source_frame_offset": 0,
            "mirror": False,
        }
    if episode_index < source_episodes + time_scaled_episodes:
        return {
            "source_dataset": source,
            "source_episode_index": episode_index - source_episodes,
            "augmentation_type": "time",
            "source_frame_stride": extraction_factor,
            "source_frame_offset": 0,
            "mirror": False,
        }
    return {
        "source_dataset": source,
        "source_episode_index": episode_index - source_episodes - time_scaled_episodes,
        "augmentation_type": "mirror",
        "source_frame_stride": 1,
        "source_frame_offset": 0,
        "mirror": True,
    }


def _episode_mapping(row: dict[str, Any], report: dict[str, Any], default_extraction_factor: int) -> dict[str, Any]:
    if "source_episode_index" in row and "augmentation_type" in row:
        return {
            "source_dataset": row.get("source_dataset") or report["source"],
            "source_episode_index": int(row["source_episode_index"]),
            "augmentation_type": row["augmentation_type"],
            "source_frame_stride": int(row.get("source_frame_stride", 1)),
            "source_frame_offset": int(row.get("source_frame_offset", 0)),
            "mirror": bool(row.get("mirror", row["augmentation_type"] == "mirror")),
        }
    return _infer_mapping_from_order(row, report, default_extraction_factor)


def _write_npz(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _log(message: str) -> None:
    print(message, flush=True)


def _source_prediction_path(cache_dir: pathlib.Path, source_episode_index: int) -> pathlib.Path:
    return cache_dir / f"episode_{source_episode_index:06d}.npz"


def _predict_source_episode(
    *,
    source: pathlib.Path,
    source_info: dict[str, Any],
    source_episode_index: int,
    cache_dir: pathlib.Path,
    config,
    model,
    device,
    prompt: str,
    batch_size: int,
    relative_interval: int,
    samples_per_batch: int,
    seed: int,
    use_cache: bool,
) -> dict[str, np.ndarray]:
    cache_path = _source_prediction_path(cache_dir, source_episode_index)
    if use_cache and cache_path.exists():
        return _load_npz(cache_path)

    parquet_path = source / _format_data_path(source_info, source_episode_index)
    episode_frame = pd.read_parquet(parquet_path)
    predictions = _predict_episode(
        src=source,
        info=source_info,
        episode_index=source_episode_index,
        df=episode_frame,
        config=config,
        model=model,
        device=device,
        prompt=prompt,
        batch_size=batch_size,
        relative_interval=relative_interval,
        samples_per_batch=samples_per_batch,
        seed=seed,
    )
    _write_npz(cache_path, predictions)
    return predictions


def _map_source_values(
    source_predictions: dict[str, np.ndarray],
    *,
    target_length: int,
    stride: int,
    offset: int,
    relative_interval: int,
) -> dict[str, np.ndarray]:
    source_length = len(source_predictions["absolute_value"])
    source_indices = offset + np.arange(target_length) * stride
    source_indices = np.minimum(source_indices, source_length - 1).astype(np.int64)
    absolute_value = source_predictions["absolute_value"][source_indices].astype(np.float32)

    absolute_advantage = np.zeros((target_length,), dtype=np.float32)
    for frame_idx in range(target_length):
        future_idx = min(frame_idx + relative_interval, target_length - 1)
        if future_idx == frame_idx:
            continue
        delta = future_idx - frame_idx
        scale = relative_interval / delta if delta != relative_interval else 1.0
        absolute_advantage[frame_idx] = (absolute_value[future_idx] - absolute_value[frame_idx]) * scale

    absolute_advantage = np.clip(absolute_advantage, -1.0, 1.0).astype(np.float32)
    return {
        "relative_advantage": absolute_advantage.copy(),
        "absolute_value": np.clip(absolute_value, -1.0, 1.0).astype(np.float32),
        "absolute_advantage": absolute_advantage,
    }


def _copy_meta_stats(
    stats_meta: dict[int, dict[str, Any]],
    old_episode_index: int,
    new_episode_index: int,
) -> dict[str, Any] | None:
    if old_episode_index not in stats_meta:
        return None
    row = dict(stats_meta[old_episode_index])
    row["episode_index"] = new_episode_index
    return row


def _write_tasks_with_prompt(dst: pathlib.Path, task: str) -> None:
    _write_tasks(dst, task)


def _count_mappings(mappings: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"original": 0, "time": 0, "mirror": 0}
    for mapping in mappings:
        key = str(mapping["augmentation_type"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_awbc_from_mapping(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    augmented = args.augmented.resolve()
    dst = args.dst.resolve()
    source_info = _load_json(source / "meta/info.json")
    augmented_info = _load_json(augmented / "meta/info.json")
    augment_report = _load_json(augmented / "augment_report.json")
    augmented_episodes = parse_episodes(args.episodes, int(augmented_info["total_episodes"]))

    augmented_episode_rows = {int(row["episode_index"]): row for row in _load_jsonl(augmented / "meta/episodes.jsonl")}
    mappings = {
        episode_index: _episode_mapping(
            augmented_episode_rows[episode_index],
            augment_report,
            args.default_extraction_factor,
        )
        for episode_index in augmented_episodes
    }
    source_episode_ids = sorted({int(mapping["source_episode_index"]) for mapping in mappings.values()})
    _log(
        "mapped "
        f"{len(augmented_episodes)} augmented episodes to {len(source_episode_ids)} source episodes; "
        f"counts={_count_mappings(mappings.values())}"
    )

    if args.dry_run:
        report = {
            "source": str(source),
            "augmented": str(augmented),
            "destination": str(dst),
            "augmented_episode_count": len(augmented_episodes),
            "source_episode_count": len(source_episode_ids),
            "source_episodes_preview": source_episode_ids[:20],
            "mapping_counts": _count_mappings(mappings.values()),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    import torch  # noqa: PLC0415

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    _log(f"loading Stage Advantage checkpoint from {args.checkpoint.resolve()} on {device}")
    config, model = _load_model(args.config_name, args.checkpoint.resolve(), device)
    _log("loaded Stage Advantage checkpoint")
    cache_dir = args.cache_dir.resolve() if args.cache_dir is not None else dst / "source_advantage_cache"

    source_predictions = {}
    _log(f"predicting {len(source_episode_ids)} source episodes")
    for source_episode_index in tqdm.tqdm(source_episode_ids, desc="Predict source episodes"):
        source_predictions[source_episode_index] = _predict_source_episode(
            source=source,
            source_info=source_info,
            source_episode_index=source_episode_index,
            cache_dir=cache_dir,
            config=config,
            model=model,
            device=device,
            prompt=args.task,
            batch_size=args.batch_size,
            relative_interval=args.relative_interval,
            samples_per_batch=args.samples_per_batch,
            seed=args.seed,
            use_cache=not args.no_cache,
        )

    _log(f"mapping predictions to {len(augmented_episodes)} augmented episodes")
    augmented_stats_path = augmented / "meta/episodes_stats.jsonl"
    augmented_stats_meta = {}
    if augmented_stats_path.exists():
        augmented_stats_meta = {int(row["episode_index"]): row for row in _load_jsonl(augmented_stats_path)}

    new_episode_rows: list[dict[str, Any]] = []
    new_stats_rows: list[dict[str, Any]] = []
    output_parquets: list[pathlib.Path] = []
    total_frames = 0
    total_videos = 0

    for new_episode_index, old_episode_index in enumerate(tqdm.tqdm(augmented_episodes, desc="Map augmented episodes")):
        mapping = mappings[old_episode_index]
        old_parquet = augmented / _format_data_path(augmented_info, old_episode_index)
        new_parquet = dst / _format_data_path(augmented_info, new_episode_index)
        episode_frame = pd.read_parquet(old_parquet).copy()
        predictions = _map_source_values(
            source_predictions[int(mapping["source_episode_index"])],
            target_length=len(episode_frame),
            stride=int(mapping["source_frame_stride"]),
            offset=int(mapping["source_frame_offset"]),
            relative_interval=args.relative_interval,
        )

        length = len(episode_frame)
        episode_frame["episode_index"] = np.full(length, new_episode_index, dtype=np.int64)
        episode_frame["frame_index"] = np.arange(length, dtype=np.int64)
        episode_frame["index"] = np.arange(total_frames, total_frames + length, dtype=np.int64)
        for key, value in predictions.items():
            episode_frame[key] = value
        new_parquet.parent.mkdir(parents=True, exist_ok=True)
        episode_frame.to_parquet(new_parquet, index=False)
        output_parquets.append(new_parquet)

        episode_row = dict(augmented_episode_rows[old_episode_index])
        episode_row.update(mapping)
        episode_row["episode_index"] = new_episode_index
        episode_row["source_augmented_episode_index"] = old_episode_index
        episode_row["tasks"] = [
            f"{args.task}, Advantage: bad",
            f"{args.task}, Advantage: neutral",
            f"{args.task}, Advantage: positive",
        ]
        episode_row["length"] = length
        new_episode_rows.append(episode_row)

        stats_row = _copy_meta_stats(augmented_stats_meta, old_episode_index, new_episode_index)
        if stats_row is not None:
            new_stats_rows.append(stats_row)

        for video_key in OPENARM_VIDEO_KEYS:
            old_video = augmented / _format_video_path(augmented_info, old_episode_index, video_key)
            new_video = dst / _format_video_path(augmented_info, new_episode_index, video_key)
            _copy_or_link(old_video, new_video, args.copy_mode)
            total_videos += 1

        total_frames += length

    discretize_report = _assign_awbc_labels(
        dst,
        output_parquets,
        bad_percentile=args.bad_percentile,
        positive_percentile=args.positive_percentile,
    )

    new_info = _add_awbc_features(augmented_info)
    new_info["total_episodes"] = len(augmented_episodes)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = total_videos
    new_info["total_chunks"] = max(1, math.ceil(len(augmented_episodes) / int(augmented_info["chunks_size"])))
    new_info["splits"] = {"train": f"0:{len(augmented_episodes)}"}
    _write_json(dst / "meta/info.json", new_info)
    _write_tasks_with_prompt(dst, args.task)
    _write_jsonl(dst / "meta/episodes.jsonl", new_episode_rows)
    if new_stats_rows:
        _write_jsonl(dst / "meta/episodes_stats.jsonl", new_stats_rows)

    report = {
        "source": str(source),
        "augmented": str(augmented),
        "destination": str(dst),
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "total_episodes": len(augmented_episodes),
        "total_source_episodes_predicted": len(source_episode_ids),
        "mapping_counts": _count_mappings(mappings.values()),
        "relative_interval": args.relative_interval,
        "samples_per_batch": args.samples_per_batch,
        "cache_dir": str(cache_dir),
        "discretize": discretize_report,
    }
    _write_json(dst / "awbc_mapping_build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True, help="Original HQ source dataset")
    parser.add_argument("--augmented", type=pathlib.Path, required=True, help="TDA augmented dataset")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="Destination AWBC dataset")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Stage Advantage checkpoint directory")
    parser.add_argument("--config-name", default="ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD")
    parser.add_argument("--episodes", default=None, help="Augmented episode spec, e.g. 0:10 or 999:1004")
    parser.add_argument("--task", default="fold the cloth")
    parser.add_argument("--relative-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-batch", type=int, default=1)
    parser.add_argument("--bad-percentile", type=float, default=20.0)
    parser.add_argument("--positive-percentile", type=float, default=70.0)
    parser.add_argument("--default-extraction-factor", type=int, default=2)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    build_awbc_from_mapping(args)


if __name__ == "__main__":
    main()
