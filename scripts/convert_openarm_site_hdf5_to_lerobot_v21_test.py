import json

import h5py
import numpy as np
import pandas as pd

from scripts import convert_openarm_site_hdf5_to_lerobot_v21 as converter


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_raw_episode(root, episode_index, length):
    hdf5_path = root / "episodes" / f"episode_{episode_index:06d}.hdf5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        f.attrs["episode_index"] = episode_index
        f.attrs["fps"] = 30
        f.attrs["num_frames"] = length
        f.attrs["raw_format_version"] = "openarm_raw_hdf5_v1"
        f.create_dataset("observation.state", data=np.full((length, 16), episode_index, dtype=np.float32))
        f.create_dataset("action", data=np.full((length, 16), episode_index + 1, dtype=np.float32))
        f.create_dataset("frame_index", data=np.arange(length, dtype=np.int64))
        f.create_dataset("timestamp", data=np.arange(length, dtype=np.float64) / 30.0)
        f.create_dataset("task", data=np.asarray([b"raw task"] * length, dtype=object))
        video_group = f.create_group("videos")
        for video_key in converter.VIDEO_KEYS:
            group = video_group.create_group(video_key)
            raw_video = root / "videos" / video_key / f"episode_{episode_index:06d}.mp4"
            raw_video.parent.mkdir(parents=True, exist_ok=True)
            raw_video.write_bytes(f"video-{video_key}-{episode_index}".encode())
            group.attrs["path"] = str(raw_video.relative_to(root))
            group.attrs["num_frames"] = length
    return hdf5_path


def test_convert_site_hdf5_to_lerobot_v21(tmp_path):
    raw = tmp_path / "ipc" / "fold_cloth1"
    dst = tmp_path / "openarm_site_align_v1"
    info = {
        "raw_format_version": "openarm_raw_hdf5_v1",
        "repo_id": "production/fold_cloth1",
        "robot_type": "openarm",
        "fps": 30,
        "vcodec": "h264",
        "schema": {
            "features": {
                "observation.images.left_wrist": {"dtype": "video", "shape": [8, 8, 3]},
                "observation.images.right_wrist": {"dtype": "video", "shape": [8, 8, 3]},
                "observation.images.base": {"dtype": "video", "shape": [4, 4, 3]},
            }
        },
        "total_episodes": 2,
        "total_frames": 11,
    }
    _write_json(raw / "meta/info.json", info)
    _write_jsonl(
        raw / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "num_frames": 5, "tasks": ["raw task"]},
            {"episode_index": 1, "num_frames": 6, "tasks": ["raw task"]},
        ],
    )
    _make_raw_episode(raw, 0, 5)
    _make_raw_episode(raw, 1, 6)

    report = converter.convert_dataset(
        [tmp_path / "ipc"],
        dst,
        dataset_id="openarm_site_align_v1",
        task=converter.DEFAULT_TASK,
        val_count=1,
        fps=30,
        chunks_size=1000,
        copy_mode="copy",
        overwrite=False,
        dry_run=False,
        episodes=None,
        max_episodes=None,
        min_frames=2,
        max_abs_state_action=None,
        timestamp_mode="raw_zeroed",
        gripper_action_fallback="state",
        verify_video_frames=False,
        skip_invalid=False,
        site_repeat=5,
    )

    out_info = json.loads((dst / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (dst / "meta/episodes.jsonl").read_text().splitlines()]
    first = pd.read_parquet(dst / "data/chunk-000/episode_000000.parquet")
    second = pd.read_parquet(dst / "data/chunk-000/episode_000001.parquet")

    assert report["valid_episodes"] == 2
    assert report["raw_joint_unit_hint"] == "radian_like"
    assert report["output_joint_unit_hint"] == "degree_like"
    assert report["policy_joint_unit"] == "degrees"
    assert report["policy_gripper_unit"] == "dataset_degrees"
    assert out_info["codebase_version"] == "v2.1"
    assert out_info["site_conversion"]["policy_joint_unit"] == "degrees"
    assert out_info["site_conversion"]["policy_gripper_unit"] == "dataset_degrees"
    assert out_info["site_conversion"]["gripper_raw_open_norm"] == converter.GRIPPER_RAW_OPEN_NORM
    assert out_info["splits"] == {"train": "0:1", "val": "1:2"}
    assert out_info["features"]["observation.images.base"]["shape"] == [4, 4, 3]
    assert episodes[0]["tasks"] == [converter.DEFAULT_TASK]
    assert episodes[1]["source_episode_index"] == 1
    assert first["episode_index"].unique().tolist() == [0]
    assert second["index"].iloc[0] == 5
    first_state = np.asarray(first["observation.state"].iloc[0])
    first_action = np.asarray(first["action"].iloc[0])
    assert first_state.shape == (16,)
    assert np.isclose(first_action[0], 180.0 / np.pi)
    assert np.isclose(first_state[7], -66.0)
    assert np.isclose(first_action[7], 0.0)
    assert (dst / "videos/chunk-000/observation.images.base/episode_000001.mp4").read_bytes().startswith(b"video-")


def test_dry_run_reports_invalid_episode(tmp_path):
    raw = tmp_path / "ipc" / "fold_cloth1"
    dst = tmp_path / "out"
    _write_json(raw / "meta/info.json", {"raw_format_version": "openarm_raw_hdf5_v1", "fps": 30})
    hdf5_path = raw / "episodes" / "episode_000000.hdf5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("observation.state", data=np.zeros((3, 16), dtype=np.float32))
        f.create_dataset("action", data=np.zeros((2, 16), dtype=np.float32))
        f.create_dataset("timestamp", data=np.arange(3, dtype=np.float64))

    report = converter.convert_dataset(
        [tmp_path / "ipc"],
        dst,
        dataset_id="site",
        task=converter.DEFAULT_TASK,
        val_count=0,
        fps=30,
        chunks_size=1000,
        copy_mode="copy",
        overwrite=False,
        dry_run=True,
        episodes=None,
        max_episodes=None,
        min_frames=2,
        max_abs_state_action=None,
        timestamp_mode="raw_zeroed",
        gripper_action_fallback="state",
        verify_video_frames=False,
        skip_invalid=False,
        site_repeat=5,
    )

    assert report["valid_episodes"] == 0
    assert report["rejected_episodes"] == 1
    assert "length mismatch" in report["rejected"][0]["reason"]
