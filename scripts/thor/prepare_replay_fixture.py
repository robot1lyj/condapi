"""Prepare local, recorded YAM observations and benchmark-only normalization.

This is numerical/performance calibration for the unfinetuned base model, not
production training statistics or evidence of robot task success.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.shared import normalize

VIEWS = {"top": "top_rgb", "left_wrist": "left_rgb", "right_wrist": "right_rgb"}
MASK = transforms.make_bool_mask(6, -1, 6, -1)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def action_chunks(state, action, start, stop, horizon=50):
    indices = np.minimum(np.arange(start, stop)[:, None] + np.arange(horizon)[None, :], len(action) - 1)
    # All future targets are relative to the CURRENT state, not pairwise deltas.
    return transforms.DeltaActions(MASK)({"state": state[start:stop], "actions": action[indices].copy()})


def read_frames(path, indices, fps):
    wanted = set(indices)
    result = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                timestamp = float(frame.pts * frame.time_base)
                if abs(timestamp - index / fps) > 1 / fps:
                    raise ValueError(f"Frame timestamp mismatch in {path}: {index}")
                result[index] = frame.to_ndarray(format="rgb24")
            if len(result) == len(wanted):
                break
    if set(result) != wanted:
        raise ValueError(f"Missing recorded frames in {path}")
    return result


def prepare(root, output, episodes):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or root == output or root in output.parents or output in root.parents:
        raise ValueError("Use a new output directory separate from the original dataset")
    manifest = root / "manifests/train.jsonl"
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [e for e in entries if e["source_episode_index"] in episodes]
    if len(selected) != len(episodes) or len(set(episodes)) != len(episodes):
        raise ValueError("Selected episode IDs must be unique and present exactly once")
    sources = {manifest}
    records = []
    for entry in selected:
        eid = entry["source_episode_index"]
        data_path = root / f"train/data/source-file-{entry['data']['file_index']:03d}.parquet"
        videos = {view: root / f"train/videos/{view}/episode-{eid:06d}.mp4" for view in VIEWS}
        sources.update([data_path, *videos.values()])
        table = pq.read_table(data_path, filters=[("episode_index", "=", eid)])
        table = table.sort_by([("frame_index", "ascending")])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        length = entry["length"]
        if state.shape != (length, 14) or action.shape != (length, 14):
            raise ValueError("YAM state/action shape mismatch")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Non-finite YAM values")
        if not np.array_equal(np.asarray(table["frame_index"]), np.arange(length)):
            raise ValueError("Non-contiguous frame indices")
        if not np.allclose(np.asarray(table["timestamp"]), np.arange(length) / entry["fps"], rtol=0, atol=1e-3):
            raise ValueError("Low-dimensional timestamps mismatch")
        if length < 200 or not entry["task"].strip():
            raise ValueError("Need a task and a sufficiently long recording")
        records.append((entry, state, action, videos))
    fingerprints = {str(p.relative_to(root)): digest(p) for p in sorted(sources)}
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for _entry, state, action, _videos in records:
        for start in range(0, len(state), 64):
            batch = action_chunks(state, action, start, min(start + 64, len(state)))
            for key, stat in stats.items():
                stat.update(batch[key])
    output.mkdir(parents=True)
    normalize.save(output, {key: stat.get_statistics() for key, stat in stats.items()})
    norm_path = output / "norm_stats.json"
    common = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_kind": "real_yam_recording",
        "source_root": str(root),
        "source_files": fingerprints,
        "norm_stats_sha256": digest(norm_path),
        "normalization_scope": "benchmark_only_same_recordings_not_production_training_stats",
        "normalization_method": "OpenPI RunningStats; 50 future actions relative to current state; endpoint clamp",
        "joint_delta_mask": list(MASK),
        "action_horizon": 50,
        "units": "original_dataset_values_no_rescaling_no_robot_calibration_claim",
        "base_model_only": True,
    }
    samples = []
    for entry, state, _action, videos in records:
        eid = entry["source_episode_index"]
        indices = [100, len(state) // 2, len(state) - 100]
        frames = {view: read_frames(path, indices, entry["fps"]) for view, path in videos.items()}
        for phase, index in zip(("early", "middle", "late"), indices, strict=True):
            name = f"episode-{eid:06d}-{phase}"
            sample_path = output / f"{name}.npz"
            np.savez_compressed(
                sample_path,
                **{
                    "observation.state": state[index],
                    "prompt": np.asarray(entry["task"]),
                    **{f"observation.images.{key}": frames[view][index] for view, key in VIEWS.items()},
                },
            )
            provenance = {
                **common,
                "episode": entry,
                "frame_index": index,
                "phase": phase,
                "sample_sha256": digest(sample_path),
            }
            (output / f"{name}.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
            samples.append({"sample": sample_path.name, "provenance": f"{name}.json"})
    if any(digest(root / relative) != sha for relative, sha in fingerprints.items()):
        raise RuntimeError("Source changed during preparation; incomplete output must not be used")
    suite = {**common, "samples": samples, "norm_stats": norm_path.name, "observation_count": len(samples)}
    (output / "suite.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(output), "samples": len(samples), "episodes": episodes}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[95, 96, 97])
    args = parser.parse_args()
    prepare(args.root, args.output, args.episodes)


if __name__ == "__main__":
    main()
