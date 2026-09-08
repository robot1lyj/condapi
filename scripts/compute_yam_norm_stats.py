"""Compute CPU-only norm stats for published, episode-per-Parquet YAM exports.

Uses every train frame (including the final partial batch), H=50 future actions
with terminal-frame repetition, and the training DeltaActions transform. Images
are irrelevant to these statistics; real LeRobot/YamInputs samples independently
verify the numerical path at start/middle/end of representative episodes.
"""

import hashlib
import json
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi import transforms
from openpi.models import model
from openpi.policies import yam_policy
from openpi.shared import normalize


def action_chunks(states: np.ndarray, actions: np.ndarray, indices: np.ndarray, horizon: int) -> dict:
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 14:
        raise ValueError("Expected matching Nx14 state/action arrays")
    if horizon < 1 or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Invalid horizon or nonfinite source")
    future = np.minimum(indices[:, None] + np.arange(horizon), len(actions) - 1)
    return transforms.DeltaActions(transforms.make_bool_mask(6, -1, 6, -1))(
        {"state": states[indices], "actions": actions[future].copy()}
    )


def read_episode(path: pathlib.Path, episode: int, length: int, fps: float):
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index", "timestamp"])
    if len(table) != length or not np.all(table["episode_index"].to_numpy() == episode):
        raise ValueError(f"Not one complete expected episode: {path}")
    np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(length))
    np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(length) / fps, atol=1e-4, rtol=1e-6)
    arrays = [np.asarray(table[k].combine_chunks().values).reshape(length, 14) for k in ("observation.state", "action")]
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError(f"Nonfinite values: {path}")
    return arrays


def verify_loader(root: pathlib.Path, episodes: list[dict], files: list[pathlib.Path], fps: float, horizon: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    checked = []
    for position in sorted({0, len(episodes) // 2, len(episodes) - 1}):
        episode = episodes[position]
        ep_id, length = episode["episode_index"], episode["length"]
        dataset = LeRobotDataset(
            "local/train",
            root=root,
            episodes=[ep_id],
            delta_timestamps={"action": [t / fps for t in range(horizon)]},
            video_backend="pyav",
        )
        states, actions = read_episode(files[position], ep_id, length, fps)
        for frame in sorted({0, length // 2, max(0, length - horizon), length - 1}):
            raw = dataset[frame]
            if int(raw["episode_index"]) != ep_id or int(raw["frame_index"]) != frame:
                raise ValueError("Loader episode/frame mismatch")
            actual = yam_policy.YamInputs(model.ModelType.PI05)(raw)
            actual = transforms.DeltaActions(transforms.make_bool_mask(6, -1, 6, -1))(actual)
            expected = action_chunks(states, actions, np.array([frame]), horizon)
            for key in ("state", "actions"):
                np.testing.assert_array_equal(actual[key], expected[key][0])
            checked.append(
                {
                    "episode": ep_id,
                    "frame": frame,
                    "state_actions_equal": True,
                    "images": {k: list(v.shape) for k, v in actual["image"].items()},
                    "task": raw.get("task"),
                }
            )
    return checked


def main(dataset: pathlib.Path, output: pathlib.Path, horizon: int = 50, block_size: int = 1024):
    if horizon != 50 or block_size < 1:
        raise ValueError("This version is gated for Pi0.5 H=50 and positive block size")
    dataset = dataset.resolve()
    if dataset.name != "train" or ".incomplete" in str(dataset):
        raise ValueError("Requires a published train split")
    manifest_path = dataset / "conversion_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    info_path = dataset / "meta/info.json"
    info_bytes = info_path.read_bytes()
    info = json.loads(info_bytes)
    if manifest["split"] != "train" or info["codebase_version"] != "v3.0" or info["robot_type"] != "yam":
        raise ValueError("Wrong dataset contract")
    if manifest["contract"]["action_mode"] != "absolute":
        raise ValueError("Delta conversion requires absolute source actions")
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [14]:
            raise ValueError("Expected 14D source")
    episodes = manifest["episodes"]
    files = sorted((dataset / "data").glob("chunk-*/file-*.parquet"))
    if len(files) != len(episodes) or len(episodes) != info["total_episodes"]:
        raise ValueError("Requires the converter's one-Parquet-per-episode layout")
    if [ep["episode_index"] for ep in episodes] != list(range(len(episodes))):
        raise ValueError("Episode order mismatch")
    if sum(ep["length"] for ep in episodes) != info["total_frames"]:
        raise ValueError("Frame count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    initial = {str(p.relative_to(dataset)): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    started = time.monotonic()
    checked = verify_loader(dataset, episodes, files, info["fps"], horizon)
    (output / "loader_equivalence.json").write_text(json.dumps(checked, indent=2) + "\n")
    print(f"LOADER_EQUIVALENCE_PASS samples={len(checked)}", flush=True)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    counts = 0
    hashes = {}
    for i, (episode, path) in enumerate(zip(episodes, files, strict=True)):
        hashes[str(path.relative_to(dataset))] = hashlib.sha256(path.read_bytes()).hexdigest()
        states, actions = read_episode(path, episode["episode_index"], episode["length"], info["fps"])
        for first in range(0, len(states), block_size):
            batch = action_chunks(states, actions, np.arange(first, min(first + block_size, len(states))), horizon)
            for key, running in stats.items():
                # Float64 accumulation avoids cancellation in long running stats.
                running.update(batch[key].astype(np.float64))
        counts += len(states)
        if i % 25 == 0 or i + 1 == len(episodes):
            progress = {
                "episodes": i + 1,
                "total_episodes": len(episodes),
                "frames": counts,
                "total_frames": info["total_frames"],
                "seconds": time.monotonic() - started,
            }
            print(json.dumps(progress), flush=True)
            (output / "progress.json").write_text(json.dumps(progress) + "\n")
    current = {str(p.relative_to(dataset)): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    if current != initial or manifest_path.read_bytes() != manifest_bytes or info_path.read_bytes() != info_bytes:
        raise ValueError("Source changed during computation")
    result = {key: value.get_statistics() for key, value in stats.items()}
    for value in result.values():
        if any(x.shape != (14,) or not np.isfinite(x).all() for x in (value.mean, value.std, value.q01, value.q99)):
            raise ValueError("Invalid statistics")
        if np.any(value.q01 > value.q99):
            raise ValueError("Inverted quantiles")
    provenance = {
        "dataset": str(dataset),
        "split": "train",
        "frames": counts,
        "episodes": len(episodes),
        "action_vectors": counts * horizon,
        "horizon": horizon,
        "block_size": block_size,
        "delta_mask": list(transforms.make_bool_mask(6, -1, 6, -1)),
        "contract": manifest["contract"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "info_sha256": hashlib.sha256(info_bytes).hexdigest(),
        "parquet_sha256": hashes,
        "quantiles": "OpenPI RunningStats 5000-bin approximate histograms; float64 accumulation",
        "tail": "repeat terminal action; include all frames",
        "validation_split_included": False,
        "seconds": time.monotonic() - started,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    temporary = output / "norm_stats.json.incomplete"
    temporary.write_text(normalize.serialize_json(result) + "\n")
    temporary.rename(output / "norm_stats.json")
    print(f"NORM_COMPLETE={output / 'norm_stats.json'} frames={counts}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
