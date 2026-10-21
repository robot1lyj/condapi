"""Extract real three-view YAM RTC replay cases from a LeRobot v3 dataset.

Reads recordings only; does not load a policy or run training. Video offsets
come from episode metadata, and checkpoint training normalization is preserved.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow.parquet as pq

from rtc_norm_identity import checkpoint_norm_identity

VIEWS = tuple(f"observation.images.{view}_rgb" for view in ("top", "left", "right"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path escapes root")
    return path


def read_rgb(path, target_times, fps):
    """Decode actual RGB frames, checking PTS against each target timestamp."""
    result = {}
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError(f"Missing video PTS: {path}")
            timestamp = float(frame.pts * frame.time_base)
            for index, target in target_times.items():
                if index not in result and abs(timestamp - target) <= 0.5 / fps + 1e-6:
                    rgb = frame.to_ndarray(format="rgb24")
                    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
                        raise ValueError("Expected native YAM 224x224 RGB; do not silently resize")
                    result[index] = rgb
            if len(result) == len(target_times):
                break
    if result.keys() != target_times.keys():
        raise ValueError(f"Video does not cover requested timestamps: {path}")
    return result


def prepare(dataset_root, checkpoint, training_contract, output, episodes):
    root, checkpoint, output = (Path(p).resolve() for p in (dataset_root, checkpoint, output))
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Use a fresh output outside the source dataset")
    if len(episodes) < 3 or len(set(episodes)) != len(episodes):
        raise ValueError("Select at least three distinct real episodes")
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info["codebase_version"] != "v3.0" or info["robot_type"] != "yam" or info["fps"] != 30:
        raise ValueError("This extractor requires LeRobot v3 YAM at 30 Hz")
    contract = json.loads(Path(training_contract).read_text())
    model = contract["model"]
    delay_max = model["rtc_training_max_delay"]
    if model["pi05"] is not True or model["action_horizon"] != 50 or not 0 < delay_max < 50:
        raise ValueError("Expected trained Pi0.5 RTC H50 contract")
    norm = checkpoint / "assets/yam/norm_stats.json"
    identity = checkpoint_norm_identity(norm, contract["norm_sha256"])
    norm_sha = digest(norm)
    metadata_path = checkpoint / "params/_METADATA"
    metadata_sha = digest(metadata_path)
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    rows = {}
    sources = {info_path, *episode_files}
    for path in episode_files:
        for row in pq.read_table(path).to_pylist():
            if row["episode_index"] in episodes:
                if row["episode_index"] in rows:
                    raise ValueError("Duplicate episode metadata")
                rows[row["episode_index"]] = row
    if set(rows) != set(episodes):
        raise ValueError("Selected episodes missing from metadata")
    output.mkdir(parents=True)
    shutil.copy2(norm, output / "norm_stats.json")
    cases, samples = [], []
    for eid in episodes:
        entry = rows[eid]
        data = checked_path(root, info["data_path"].format(
            chunk_index=entry["data/chunk_index"], file_index=entry["data/file_index"]))
        sources.add(data)
        table = pq.read_table(data, filters=[("episode_index", "=", eid)]).sort_by([("frame_index", "ascending")])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(table["timestamp"], dtype=np.float64)
        length = entry["length"]
        if length < 201 or state.shape != (length, 14) or action.shape != (length, 14):
            raise ValueError("Invalid episode lengths or YAM 14D contract")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Nonfinite source state/actions")
        if not np.array_equal(np.asarray(table["frame_index"]), np.arange(length)):
            raise ValueError("Noncontiguous episode frames")
        if not np.allclose(timestamps, np.arange(length) / 30, rtol=0, atol=1e-3):
            raise ValueError("Episode timestamps do not match 30 Hz rows")
        prompt = entry["tasks"]
        if len(prompt) != 1 or not prompt[0].strip():
            raise ValueError("Expected exactly one episode task")
        indices = (100, length // 2, length - 100)
        images = {}
        episode_sources = {data}
        for key in VIEWS:
            prefix = f"videos/{key}"
            video = checked_path(root, info["video_path"].format(
                video_key=key, chunk_index=entry[f"{prefix}/chunk_index"],
                file_index=entry[f"{prefix}/file_index"]))
            episode_sources.add(video)
            sources.add(video)
            offset = entry[f"{prefix}/from_timestamp"]
            targets = {index: offset + index / 30 for index in indices}
            if max(targets.values()) >= entry[f"{prefix}/to_timestamp"]:
                raise ValueError("Video segment does not cover requested frames")
            images[key] = read_rgb(video, targets, 30)
        fingerprints = {str(p.relative_to(root)): digest(p) for p in sorted(episode_sources)}
        for index, phase, delay in zip(indices, ("early", "middle", "late"), (0, 1, delay_max), strict=True):
            name = f"episode-{eid:06d}-{phase}"
            sample = output / f"{name}.npz"
            np.savez_compressed(sample, **{"observation.state": state[index],
                "prompt": np.asarray(prompt[0]), **{key: images[key][index] for key in VIEWS}})
            prefix_path = output / f"{name}-prefix.npy"
            np.save(prefix_path, action[index:index + delay].copy(), allow_pickle=False)
            provenance = dict(source_kind="real_yam_recording", source_root=str(root),
                source_files=fingerprints, source_split=root.name,
                episode={k: v for k, v in entry.items() if not k.startswith("stats/")},
                frame_index=index, phase=phase, sample_sha256=digest(sample),
                norm_stats_sha256=norm_sha, checkpoint_metadata_sha256=metadata_sha,
                normalization_scope="checkpoint_training_norm")
            (output / f"{name}.json").write_text(json.dumps(provenance, indent=2) + "\n")
            samples.append(dict(sample=sample.name, provenance=f"{name}.json"))
            cases.append(dict(**samples[-1], source_episode=str(data.relative_to(root)),
                committed_actions=prefix_path.name, committed_actions_sha256=digest(prefix_path),
                delay_steps=delay, observation_policy_tick=index, target_start_tick=index,
                committed_start_tick=index, observation_dataset_timestamp_s=float(timestamps[index]),
                action_0_dataset_timestamp_s=float(timestamps[index]),
                timing_scope="offline dataset policy tick, not live exposure/control offset"))
    common = dict(source_kind="real_yam_recording", source_root=str(root),
        norm_stats_sha256=norm_sha, norm_identity=identity,
        checkpoint_metadata_sha256=metadata_sha,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        validation_scope="real recorded observations/prefixes; not closed-loop task success")
    (output / "cases.json").write_text(json.dumps(dict(**common, cases=cases), indent=2) + "\n")
    (output / "suite.json").write_text(json.dumps(dict(**common, samples=samples,
        source_files={str(p.relative_to(root)): digest(p) for p in sorted(sources)},
        base_model_only=False, observation_count=len(samples)), indent=2) + "\n")
    print(json.dumps(dict(output=str(output), cases=len(cases), episodes=episodes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    prepare(args.dataset_root, args.checkpoint, args.training_contract, args.output, args.episodes)
