"""Convert a frozen ABC task subset into a new LeRobot v3 dataset, one split at a time.

Requires an explicit, evidence-backed robot contract. No source values are
rescaled and no frames are dropped. Failures leave an unpublished .incomplete
directory for inspection. The default conversion requires every selected episode.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import av
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__:
    from .audit_yam_subset import signature
    from .audit_yam_subset import validate_rows
    from .audit_yam_subset import validate_video
else:
    from audit_yam_subset import signature
    from audit_yam_subset import validate_rows
    from audit_yam_subset import validate_video

CAMERA_KEYS = {
    "top": "observation.images.top_rgb",
    "left_wrist": "observation.images.left_rgb",
    "right_wrist": "observation.images.right_rgb",
}
JOINT_NAMES = [
    *[f"left_j{i}" for i in range(1, 7)],
    "left_gripper",
    *[f"right_j{i}" for i in range(1, 7)],
    "right_gripper",
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_contract(contract):
    if contract.get("state_action_names") != JOINT_NAMES:
        raise ValueError("Contract must explicitly confirm the YAM left/right 14D layout")
    for key in ("joint_unit", "gripper_unit", "evidence"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ValueError(f"Contract requires {key}")
    if contract.get("action_mode") not in ("absolute", "delta"):
        raise ValueError("Contract action_mode must be absolute or delta")
    return contract


def source_paths(root, split, episode):
    eid = episode["source_episode_index"]
    file_id = episode["data"]["file_index"]
    if not isinstance(eid, int) or eid < 0 or not isinstance(file_id, int) or file_id < 0:
        raise ValueError("Source indices must be nonnegative integers")
    data = root / split / "data" / f"source-file-{file_id:03d}.parquet"
    videos = {key: root / split / "videos" / cam / f"episode-{eid:06d}.mp4" for cam, key in CAMERA_KEYS.items()}
    return data, videos


def convert(root, output, split, contract, episode_ids=None, min_age=300):
    root, output = Path(root).resolve(), Path(output).resolve()
    work = output.with_name(output.name + ".incomplete")
    if split not in ("train", "val") or min_age < 0:
        raise ValueError("Invalid split or minimum file age")
    validate_contract(contract)
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output must be separate from the source tree")
    if output.exists() or work.exists():
        raise FileExistsError("Output or .incomplete directory exists; use a new version")
    manifest = root / "manifests" / f"{split}.jsonl"
    content = manifest.read_bytes()
    manifest_hash = hashlib.sha256(content).hexdigest()
    all_episodes = [json.loads(line) for line in content.splitlines() if line.strip()]
    all_ids = [e["source_episode_index"] for e in all_episodes]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate episode IDs in source manifest")
    selected_ids = set(all_ids if episode_ids is None else episode_ids)
    if not selected_ids or not selected_ids.issubset(all_ids):
        raise ValueError("Requested episodes must be a nonempty subset of this split")
    episodes = [e for e in all_episodes if e["source_episode_index"] in selected_ids]
    fps_set = {e["fps"] for e in episodes}
    if len(fps_set) != 1 or next(iter(fps_set)) <= 0:
        raise ValueError("Mixed or invalid FPS")
    fps = next(iter(fps_set))
    if int(fps) != fps:
        raise ValueError("Writer requires integral FPS")
    # Freeze file identities before creating any output. Missing files stop the
    # conversion rather than silently biasing the selected split.
    snapshots = {}
    for episode in episodes:
        data, videos = source_paths(root, split, episode)
        for path in (data, *videos.values()):
            stat = signature(path)
            if stat[0] == 0 or time.time() - stat[1] / 1e9 < min_age:
                raise ValueError(f"Pending upload: {path}")
            snapshots[path] = stat
    hashes = {str(path.relative_to(root)): digest(path) for path in snapshots}
    features = {
        key: {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES} for key in ("observation.state", "action")
    }
    _, first_videos = source_paths(root, split, episodes[0])
    for key, path in first_videos.items():
        with av.open(str(path)) as container:
            frame = next(container.decode(video=0))
            features[key] = {
                "dtype": "video",
                "shape": (frame.height, frame.width, 3),
                "names": ["height", "width", "channels"],
            }
    writer = LeRobotDataset.create(
        repo_id=f"local/{output.name}",
        root=work,
        robot_type="yam",
        fps=int(fps),
        features=features,
        video_backend="pyav",
        vcodec="h264",
        streaming_encoding=False,
        encoder_threads=1,
    )
    provenance = {
        "source_root": str(root),
        "split": split,
        "source_manifest_sha256": manifest_hash,
        "contract": contract,
        "source_files_sha256": hashes,
        "subset": len(episodes) != len(all_episodes),
        "episodes": [],
        "video_encoding": "LeRobot 0.5.1 h264/yuv420p defaults; re-encoded from temporary PNGs",
        "norm_stats": "not_computed",
        "training_verified": False,
    }
    try:
        cached_path, table = None, None
        for new_id, episode in enumerate(episodes):
            data, videos = source_paths(root, split, episode)
            if cached_path != data:
                table = pq.read_table(data)
                cached_path = data
            rows = table.filter(pc.equal(table["episode_index"], episode["source_episode_index"]))
            validate_rows(rows, episode)
            for path in videos.values():
                validate_video(path, episode["length"], fps)
            states = np.asarray(rows["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(rows["action"].to_pylist(), dtype=np.float32)
            with ExitStack() as stack:
                decoders = {
                    key: stack.enter_context(av.open(str(path))).decode(video=0) for key, path in videos.items()
                }
                for i in range(episode["length"]):
                    frame = {"observation.state": states[i], "action": actions[i], "task": episode["task"]}
                    for key, decoder in decoders.items():
                        frame[key] = next(decoder).to_ndarray(format="rgb24")
                        if frame[key].shape != features[key]["shape"]:
                            raise ValueError(f"Video shape changed: {key}")
                    writer.add_frame(frame)
            writer.save_episode(parallel_encoding=False)
            provenance["episodes"].append(
                {
                    "episode_index": new_id,
                    "source_episode_index": episode["source_episode_index"],
                    "source_repo": episode["source_repo"],
                    "source_revision": episode["source_revision"],
                    "length": episode["length"],
                    "task": episode["task"],
                }
            )
            print(f"Converted {split} source={episode['source_episode_index']} -> {new_id}", flush=True)
        writer.finalize()
        # Full source digest recheck prevents publishing a version from files that
        # were replaced while the writer was active.
        if digest(manifest) != manifest_hash or any(
            signature(path) != stat or digest(path) != hashes[str(path.relative_to(root))]
            for path, stat in snapshots.items()
        ):
            raise ValueError("Source changed during conversion")
        check = LeRobotDataset(f"local/{output.name}", root=work, video_backend="pyav")
        offset = 0
        for episode in episodes:
            for index in {offset, offset + episode["length"] // 2, offset + episode["length"] - 1}:
                sample = check[index]
                if sample["observation.state"].shape != (14,) or sample["action"].shape != (14,):
                    raise ValueError("Output loader shape mismatch")
                if sample["task"] != episode["task"]:
                    raise ValueError("Output task mismatch")
            offset += episode["length"]
        if len(check) != offset or check.num_episodes != len(episodes):
            raise ValueError("Output episode/frame count mismatch")
        (work / "conversion_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        # No overwrite; a concurrent publisher must not replace an existing version.
        if output.exists():
            raise FileExistsError(output)
        work.rename(output)
    except BaseException:
        writer.finalize()
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+")
    parser.add_argument("--min-age", type=float, default=300)
    args = vars(parser.parse_args())
    args["contract"] = json.loads(args["contract"].read_text())
    print(convert(**args))


if __name__ == "__main__":
    main()
