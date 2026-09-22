"""Read-only YAM LeRobot v2 windows using native OpenWAM transforms and video IO."""

from bisect import bisect_right
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.transforms.normalize import YAML_TO_NORM_MODE
from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.utils.unify_action import map_to_unify
from openwam.dataloader.utils.video_io import decode_video_frames
import pyarrow.parquet as pq
import torch

from adapters.openwam.common import CAMERAS
from adapters.openwam.common import dataset_path
from adapters.openwam.common import read_info


def read_episode(root, info, episode):
    path = dataset_path(root, info, "data", episode)
    table = pq.read_table(path)
    values = table.to_pydict()
    if not values["episode_index"] or set(values["episode_index"]) != {episode}:
        raise ValueError(f"Episode identity mismatch: {path}")
    if values["frame_index"] != list(range(len(table))):
        raise ValueError(f"Non-contiguous frame indices: {path}")
    times = np.asarray(values["timestamp"], dtype=np.float64)
    if not np.allclose(times, np.arange(len(table)) / info["fps"], atol=1e-3):
        raise ValueError(f"Timestamp/fps mismatch: {path}")
    for key in ("action", "observation.state"):
        array = np.asarray(values[key], dtype=np.float32)
        if array.shape != (len(table), 14) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key}: {path}")
        values[key] = array
    return values


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_stats(root, episodes, output):
    """Streaming moments/extrema of the explicit train split; no video decoding or source writes."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("Statistics require a new path outside the source dataset")
    if not episodes or len(set(episodes)) != len(episodes) or any(type(e) is not int or e < 0 for e in episodes):
        raise ValueError("Select unique nonnegative training episode ids")
    info = read_info(root)
    accum = {
        key: [0, np.zeros(14), np.zeros(14), np.full(14, np.inf), np.full(14, -np.inf)]
        for key in ("action", "observation.state")
    }
    hashes = {}
    for episode in episodes:
        values = read_episode(root, info, episode)
        hashes[str(episode)] = digest(dataset_path(root, info, "data", episode))
        for key, (count, mean, m2, lo, hi) in accum.items():
            x = values[key].astype(np.float64)
            n, batch_mean = len(x), x.mean(axis=0)
            delta = batch_mean - mean
            total = count + n
            accum[key] = [
                total,
                mean + delta * n / total,
                m2 + ((x - batch_mean) ** 2).sum(axis=0) + delta**2 * count * n / total,
                np.minimum(lo, x.min(axis=0)),
                np.maximum(hi, x.max(axis=0)),
            ]
    stats = {}
    for key, (count, mean, m2, lo, hi) in accum.items():
        stats["joint" if key == "action" else "joint_state"] = {
            "mean": mean.tolist(),
            "std": np.sqrt(m2 / count).tolist(),
            "min": lo.tolist(),
            "max": hi.tolist(),
        }
    payload = {
        "schema_version": 1,
        "episodes": list(episodes),
        "fps": info["fps"],
        "info_sha256": digest(root / "meta/info.json"),
        "metadata_sha256": {
            name: digest(root / "meta" / name) for name in ("info.json", "episodes.jsonl", "tasks.jsonl")
        },
        "parquet_sha256": hashes,
        "stats": stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return payload


def load_stats(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported YAM statistics schema")
    for name in ("joint", "joint_state"):
        for key in ("mean", "std", "min", "max"):
            x = np.asarray(payload["stats"][name][key], dtype=np.float32)
            if x.shape != (14,) or not np.isfinite(x).all():
                raise ValueError(f"Invalid YAM stats {name}.{key}")
        s = payload["stats"][name]
        if np.any(np.asarray(s["std"]) < 0) or np.any(np.asarray(s["min"]) > s["max"]):
            raise ValueError("Invalid statistics bounds")
    return payload


class YamDataset(BaseDataset):
    def __init__(self, config):
        self.config = config
        self.root = Path(config["dataset_dir"]).resolve()
        self.info = read_info(self.root)
        self.episodes = list(config["episodes"])
        payload = load_stats(config["normalization_json"])
        if payload["episodes"] != self.episodes or payload["info_sha256"] != digest(self.root / "meta/info.json"):
            raise ValueError("Statistics do not match selected training data")
        for name, expected in payload["metadata_sha256"].items():
            if (
                name not in ("info.json", "episodes.jsonl", "tasks.jsonl")
                or digest(self.root / "meta" / name) != expected
            ):
                raise ValueError(f"Dataset metadata changed after statistics audit: {name}")
        if set(payload["metadata_sha256"]) != {"info.json", "episodes.jsonl", "tasks.jsonl"}:
            raise ValueError("Incomplete metadata provenance")
        self.ends = [0]
        metadata = [json.loads(line) for line in (self.root / "meta/episodes.jsonl").read_text().splitlines() if line]
        lengths = {row["episode_index"]: row["length"] for row in metadata}
        for episode in self.episodes:
            if payload["parquet_sha256"][str(episode)] != digest(dataset_path(self.root, self.info, "data", episode)):
                raise ValueError(f"Dataset changed after statistics audit: episode {episode}")
            if lengths[episode] < 2:
                raise ValueError("Training episodes must have at least two frames")
            self.ends.append(self.ends[-1] + lengths[episode] - 1)
        self.tasks = {
            row["task_index"]: row["task"]
            for row in (json.loads(line) for line in (self.root / "meta/tasks.jsonl").read_text().splitlines() if line)
        }
        self.action_norm = Normalizer(YAML_TO_NORM_MODE[config["normalize_mode"]], payload["stats"]["joint"])
        self.state_norm = Normalizer(YAML_TO_NORM_MODE[config["normalize_mode"]], payload["stats"]["joint_state"])
        self.normalization_stats_path = config["normalization_stats_path"]
        self.slots = np.asarray(config["unify_action_map"], dtype=np.int64) if config["unify_action"] else None
        self.action_dim = 80 if self.slots is not None else 14

    @classmethod
    def from_config(cls, config, split="train"):
        if split != "train":
            raise ValueError("Validation requires its own held-out configuration and training statistics")
        return cls(config)

    def __len__(self):
        return self.ends[-1]

    @functools.lru_cache(maxsize=2)  # noqa: B019 - bounded cache in the lifetime of a worker process
    def _episode(self, episode):
        values = read_episode(self.root, self.info, episode)
        local = self.episodes.index(episode)
        if len(values["frame_index"]) != self.ends[local + 1] - self.ends[local] + 1:
            raise ValueError("Episode metadata length mismatch")
        return values

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        local = bisect_right(self.ends, index) - 1
        episode, offset = self.episodes[local], index - self.ends[local]
        values = self._episode(episode)
        cfg, length = self.config, len(values["frame_index"])
        horizon = cfg["num_frames"] - 1
        count = min(horizon, length - offset)
        action = np.zeros((horizon, 14), dtype=np.float32)
        action[:count] = self.action_norm.normalize(values["action"][offset : offset + count])
        state = self.state_norm.normalize(values["observation.state"][offset : offset + 1])
        action_mask = np.broadcast_to((np.arange(horizon) < count)[:, None], (horizon, 14)).copy()
        state_mask = np.ones((1, 14), dtype=bool)
        if self.slots is not None:
            action, _ = map_to_unify(action, self.slots, 80)
            state, _ = map_to_unify(state, self.slots, 80)
            mapped_mask = np.zeros((horizon, 80), dtype=bool)
            mapped_mask[:, self.slots] = action_mask
            action_mask = mapped_mask
            state_mask = np.zeros((1, 80), dtype=bool)
            state_mask[:, self.slots] = True
        indices = np.arange(0, cfg["num_frames"], cfg["video_stride"]) + offset
        video_mask = indices < length
        indices = np.minimum(indices, length - 1).tolist()
        views = {
            camera: decode_video_frames(
                str(dataset_path(self.root, self.info, "video", episode, camera)), indices, cfg["height"], cfg["width"]
            )
            for camera in CAMERAS
        }
        video = [compose_image({camera: views[camera][i] for camera in CAMERAS}, cfg) for i in range(len(indices))]
        prompt = self.tasks[values["task_index"][offset]]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Missing task prompt")
        return {
            "video": video,
            "first_frame_image": [video[0]],
            "vace_video": None,
            "prompt": prompt,
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "proprio": torch.from_numpy(state),
            "proprio_mask": torch.from_numpy(state_mask),
            "video_mask": torch.from_numpy(video_mask),
        }


def compose_image(images, config):
    if set(images) != set(CAMERAS) or any(image.mode != "RGB" for image in images.values()):
        raise ValueError("All three YAM RGB images are required")
    return assemble_multiview_layout(images, CAMERAS, config["height"], config["width"])
