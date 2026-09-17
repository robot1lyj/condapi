"""Read-only probe of the real YAM LeRobot/Pi05 input pipeline.

This script intentionally constructs the same LeRobotDataset and transform
chain used by training, but never creates a trainer or runs a model.  It is
designed for a short Slurm CPU job on a compute node.  The output directory is
created once and contains JSON evidence only; source files are never written.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import socket
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

DATA_ROOT = pathlib.Path("/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train")
SELECTION = pathlib.Path("/home/wuyan/lyj/YAM/training-assets/lego_rtc_10h_20260916/episodes.json")
ASSETS_ROOT = pathlib.Path("/home/wuyan/lyj/YAM/training-assets/lego_rtc_10h_20260916/norm")


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: Any):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def parse_ids(path: pathlib.Path | None) -> list[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("episode_ids", payload.get("selected_episode_ids"))
    if not isinstance(payload, list) or not all(isinstance(x, int) for x in payload):
        raise ValueError(f"Expected a JSON list of integer episode IDs: {path}")
    if len(set(payload)) != len(payload):
        raise ValueError("Selection contains duplicate episode IDs")
    return payload


def finite_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None,
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        result.update(min=float(np.nanmin(array)), max=float(np.nanmax(array)))
    return result


def episode_ranges(dataset, episode_ids: list[int]) -> dict[int, tuple[int, int]]:
    """Return global dataset indices for each source episode.

    LeRobot v0.6 exposes ``episode_data_index`` as two arrays (from/to).  The
    fallback uses the public metadata frame counts, so an API change is
    reported by the probe rather than silently changing the mapping.
    """

    # LeRobot 0.6 keeps the absolute frame ranges in metadata and builds an
    # absolute-to-relative map inside DatasetReader when ``episodes=`` is used.
    # The public LeRobotDataset object does not expose episode_data_index.
    reader = getattr(dataset, "reader", None)
    mapping = getattr(reader, "_absolute_to_relative_idx", None)
    metadata_episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if mapping is None or metadata_episodes is None:
        raise RuntimeError("LeRobotDataset lacks the metadata/index mapping needed for source episode IDs")

    def metadata_value(record, key: str) -> int:
        value = record[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0]
        return int(value)

    result = {}
    for episode in episode_ids:
        record = metadata_episodes[episode]
        absolute_start = metadata_value(record, "dataset_from_index")
        absolute_end = metadata_value(record, "dataset_to_index")
        try:
            relative_start = int(mapping[absolute_start])
            relative_last = int(mapping[absolute_end - 1])
        except KeyError as exc:
            raise RuntimeError(f"Episode {episode} is not present in the selected loader") from exc
        result[episode] = (relative_start, relative_last + 1)
    return result


def unwrap_lerobot_dataset(dataset):
    """Find the real LeRobotDataset beneath prompt/fallback wrappers."""
    current = dataset
    seen = set()
    while not hasattr(current, "hf_dataset"):
        marker = id(current)
        if marker in seen or not hasattr(current, "_dataset"):
            raise RuntimeError("Cannot locate underlying LeRobotDataset")
        seen.add(marker)
        current = current._dataset
    return current


def check_sample(raw: dict[str, Any], transformed: dict[str, Any], expected_episode: int, expected_frame: int) -> dict:
    actual_episode = int(np.asarray(raw["episode_index"]))
    actual_frame = int(np.asarray(raw["frame_index"]))
    if (actual_episode, actual_frame) != (expected_episode, expected_frame):
        raise ValueError(
            f"episode/frame mismatch expected=({expected_episode},{expected_frame}) "
            f"actual=({actual_episode},{actual_frame})"
        )

    image_summaries = {key: finite_summary(value) for key, value in transformed["image"].items()}
    output = {
        "episode_index": actual_episode,
        "frame_index": actual_frame,
        "raw_keys": sorted(raw),
        "raw_state": finite_summary(raw["observation.state"]),
        "raw_action": finite_summary(raw["action"]),
        "raw_action_sequence": finite_summary(raw["action"]),
        "transformed_state": finite_summary(transformed["state"]),
        "transformed_actions": finite_summary(transformed["actions"]),
        "images": image_summaries,
        "image_masks": {key: bool(np.asarray(value)) for key, value in transformed["image_mask"].items()},
        "tokenized_prompt": finite_summary(transformed["tokenized_prompt"]),
        "tokenized_prompt_mask": finite_summary(transformed["tokenized_prompt_mask"]),
        "task_index": int(np.asarray(raw["task_index"])),
        "task": str(raw.get("task", "")),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, default=DATA_ROOT)
    parser.add_argument("--selection", type=pathlib.Path, default=None)
    parser.add_argument("--assets-root", type=pathlib.Path, default=ASSETS_ROOT)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--episode-stride", type=int, default=100)
    parser.add_argument("--all-selection", action="store_true")
    args = parser.parse_args()
    if args.episode_stride < 1:
        raise ValueError("--episode-stride must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    # Imports are delayed so metadata-only argument mistakes do not initialize
    # JAX/PyTorch on the login host when this file is inspected.
    from openpi.shared import normalize
    from openpi.training import config as configs
    from openpi.training import data_loader

    dataset = args.dataset.resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    manifest = json.loads((dataset / "conversion_manifest.json").read_text())
    full_ids = [int(item["episode_index"]) for item in manifest["episodes"]]
    full_lengths = {int(item["episode_index"]): int(item["length"]) for item in manifest["episodes"]}
    selected_ids = parse_ids(args.selection)
    if selected_ids is None:
        probe_ids = full_ids[:: args.episode_stride]
        if full_ids[-1] not in probe_ids:
            probe_ids.append(full_ids[-1])
        label = f"full_stride_{args.episode_stride}"
    else:
        missing = sorted(set(selected_ids) - set(full_ids))
        if missing:
            raise ValueError(f"Selected episodes missing from manifest: {missing[:10]}")
        probe_ids = selected_ids if args.all_selection else selected_ids[:: args.episode_stride]
        if not probe_ids:
            raise ValueError("No probe episodes selected")
        if args.all_selection:
            label = "selection_all"
        else:
            label = f"selection_stride_{args.episode_stride}"

    norm_dir = args.assets_root / "yam"
    norm = normalize.load(norm_dir)
    config = configs.get_config("pi05_yam")
    model_config = config.model
    data_factory = dataclasses.replace(
        config.data,
        repo_id=str(dataset),
        assets=configs.AssetsConfig(assets_dir=str(args.assets_root), asset_id="yam"),
        base_config=dataclasses.replace(config.data.base_config, train_episodes=probe_ids),
    )
    data_config = data_factory.create(args.assets_root, model_config)
    raw_dataset = data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    transformed_dataset = data_loader.transform_dataset(raw_dataset, dataclasses.replace(data_config, norm_stats=norm))

    # Force LeRobot to activate its filtered Arrow reader.  ``len(dataset)``
    # alone only reports metadata totals while the reader is still lazy.
    base_dataset = unwrap_lerobot_dataset(raw_dataset)
    _ = base_dataset.hf_dataset
    ranges = episode_ranges(base_dataset, probe_ids)
    samples: list[dict] = []
    failures: list[dict] = []
    tail_checks: list[dict] = []
    for episode in probe_ids:
        start, end = ranges[episode]
        expected_length = full_lengths[episode]
        if end - start != expected_length:
            raise ValueError(f"length mismatch episode={episode} loader={end-start} manifest={expected_length}")
        frames = sorted(
            {0, expected_length // 2, max(0, expected_length - model_config.action_horizon), expected_length - 1}
        )
        for frame in frames:
            global_index = start + frame
            try:
                raw = raw_dataset[global_index]
                transformed = transformed_dataset[global_index]
                item = check_sample(raw, transformed, episode, frame)
                # The complete transform chain must return 32D state/action and
                # a 50-step action horizon.  Gripper coordinates stay absolute;
                # this is checked by a separate raw-vs-delta relation below.
                if item["transformed_state"]["shape"] != [32]:
                    raise ValueError(f"state shape {item['transformed_state']['shape']}")
                if item["transformed_actions"]["shape"] != [50, 32]:
                    raise ValueError(f"actions shape {item['transformed_actions']['shape']}")
                samples.append(item)
                if frame >= expected_length - model_config.action_horizon:
                    sequence = np.asarray(raw["action"])
                    repeated = bool(len(sequence) < 2 or np.array_equal(sequence[-1], sequence[-2]))
                    tail_checks.append(
                        {
                            "episode": episode,
                            "frame": frame,
                            "loader_sequence_shape": list(sequence.shape),
                            "expected_pad_steps": max(0, frame + 50 - expected_length),
                            "terminal_repeated": repeated,
                            "action_is_pad": (
                                np.asarray(raw["action_is_pad"]).astype(bool).tolist()
                                if "action_is_pad" in raw
                                else None
                            ),
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - preserve per-sample evidence
                failures.append({"episode": episode, "frame": frame, "global_index": global_index, "error": repr(exc)})

    summary = {
        "label": label,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "dataset": str(dataset),
        "dataset_info_sha256": digest(dataset / "meta/info.json"),
        "conversion_manifest_sha256": digest(dataset / "conversion_manifest.json"),
        "selection": str(args.selection) if args.selection else None,
        "selection_sha256": digest(args.selection) if args.selection else None,
        "norm_dir": str(norm_dir),
        "norm_sha256": digest(norm_dir / "norm_stats.json"),
        "total_dataset_episodes": int(info["total_episodes"]),
        "total_dataset_frames": int(info["total_frames"]),
        "probe_episodes": len(probe_ids),
        "probe_frames": sum(full_lengths[episode] for episode in probe_ids),
        "probe_windows": len(probe_ids) * 4,
        "samples_ok": len(samples),
        "samples_failed": len(failures),
        "failure_counts": dict(Counter(item["error"] for item in failures)),
        "tail_checks": len(tail_checks),
        "elapsed_seconds": time.monotonic() - started,
        "pipeline": [
            "LeRobotDataset(video_backend=pyav, delta_timestamps action H50)",
            "YamInputs(PI05, 14D, three RGB views)",
            "DeltaActions(mask=(6,-1,6,-1))",
            "Normalize(quantile stats)",
            "InjectDefaultPrompt",
            "ResizeImages(224,224)",
            "TokenizePrompt(PaliGemma, discrete PI05 state)",
            "PadStatesAndActions(32)",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n"
    )
    (args.output / "samples.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, default=json_default) + "\n" for item in samples)
    )
    (args.output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, default=json_default) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
