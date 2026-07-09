import json

import cv2
import h5py
import numpy as np
import pandas as pd

from scripts import convert_openarm_hil_hdf5_to_lerobot_v21 as converter


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_mp4(path, frames, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _read_frame_count(path):
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _make_hil_episode(root, episode_index=0):
    length = 5
    hdf5_path = root / "episodes" / f"episode_{episode_index:06d}.hdf5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    states = np.arange(length * 16, dtype=np.float32).reshape(length, 16)
    actions = states + 100.0
    policy_actions = actions + 10.0
    teleop_actions = np.full((length, 16), np.nan, dtype=np.float32)
    teleop_actions[2] = actions[2] + 20.0
    session_state = np.asarray(["policy", "intervention_hold", "human", "intervention_hold", "policy"], dtype=object)
    selected_source = np.asarray(["policy", "hold", "human", "hold", "policy"], dtype=object)
    authority_source = np.asarray(["policy", "scripted", "human", "scripted", "policy"], dtype=object)

    with h5py.File(hdf5_path, "w") as f:
        f.attrs["episode_index"] = episode_index
        f.attrs["num_frames"] = length
        f.create_dataset("timestamp", data=np.arange(length, dtype=np.float64) / 30.0)
        f.create_dataset("timestamp_ns", data=np.arange(length, dtype=np.int64) * 33_333_333)
        f.create_dataset("episode_index", data=np.full(length, episode_index, dtype=np.int64))
        f.create_dataset("frame_index", data=np.arange(length, dtype=np.int64))
        f.create_dataset(
            "task",
            data=np.asarray(["Fold the T-shirt properly"] * length, dtype=object),
            dtype=string_dtype,
        )
        f.create_dataset(
            "prompt",
            data=np.asarray(["Fold the T-shirt properly"] * length, dtype=object),
            dtype=string_dtype,
        )
        f.create_dataset("observation.state", data=states)
        f.create_dataset("action", data=actions)
        f.create_dataset("action.executed", data=actions)
        f.create_dataset("executed_action", data=actions)
        f.create_dataset("policy_action", data=policy_actions)
        f.create_dataset("action.policy", data=policy_actions)
        f.create_dataset("teleop_action", data=teleop_actions)
        f.create_dataset("human_action", data=teleop_actions)
        f.create_dataset("action.human", data=teleop_actions)
        f.create_dataset("session_state", data=session_state, dtype=string_dtype)
        f.create_dataset("selected_source", data=selected_source, dtype=string_dtype)
        f.create_dataset("authority_source", data=authority_source, dtype=string_dtype)
        f.create_dataset("complementary_info.is_intervention", data=np.asarray([0, 1, 1, 1, 0], dtype=np.int64))
        video_group = f.create_group("videos")
        for video_key in converter.VIDEO_KEYS:
            group = video_group.create_group(video_key)
            raw_video = root / "videos" / video_key / f"episode_{episode_index:06d}.mp4"
            frames = [
                np.full((8, 10, 3), fill_value=frame_idx * 30, dtype=np.uint8)
                for frame_idx in range(length)
            ]
            _write_mp4(raw_video, frames)
            group.attrs["path"] = str(raw_video.relative_to(root))
            group.attrs["num_frames"] = length
    return hdf5_path


def test_convert_hil_hdf5_to_evo_clean_lerobot_v21(tmp_path):
    raw = tmp_path / "openarm_hil_dagger"
    dst = tmp_path / "openarm_hil_evo_v1"
    info = {
        "raw_format_version": "openarm_hil_dagger_v1",
        "repo_id": "local/openarm_hil_raw",
        "robot_type": "openarm",
        "fps": 30,
        "schema": {
            "features": {
                "observation.images.left_wrist": {"dtype": "video", "shape": [8, 10, 3]},
                "observation.images.right_wrist": {"dtype": "video", "shape": [8, 10, 3]},
                "observation.images.base": {"dtype": "video", "shape": [8, 10, 3]},
            }
        },
        "total_episodes": 1,
        "total_frames": 5,
    }
    _write_json(raw / "meta/info.json", info)
    _write_jsonl(
        raw / "meta/episodes.jsonl",
        [
            {
                "episode_index": 0,
                "num_frames": 5,
                "tasks": ["Fold the T-shirt properly"],
                "episode_success": "success",
                "episode_outcome": "success",
                "recovery_success": True,
                "collector_policy_id": "site_deg_hq5k_4999",
                "model_metadata": {"checkpoint": "4999"},
            }
        ],
    )
    _make_hil_episode(raw, 0)

    report = converter.convert_dataset(
        [raw],
        dst,
        dataset_id="openarm_hil_evo_v1",
        task=converter.DEFAULT_TASK,
        val_count=0,
        fps=30,
        chunks_size=1000,
        video_codec="mp4v",
        overwrite=False,
        dry_run=False,
        episodes=None,
        max_episodes=None,
        min_frames=2,
        timestamp_mode="fps",
        verify_video_frames=True,
        skip_invalid=False,
        default_success="failure",
    )

    frame = pd.read_parquet(dst / "data/chunk-000/episode_000000.parquet")
    episodes = [json.loads(line) for line in (dst / "meta/episodes.jsonl").read_text().splitlines()]
    info = json.loads((dst / "meta/info.json").read_text())

    assert report["valid_episodes"] == 1
    assert report["raw_total_frames"] == 5
    assert report["total_frames"] == 3
    assert report["policy_frames"] == 2
    assert report["human_frames"] == 1
    assert report["dropped_hold_frames"] == 2
    assert frame["frame_index"].tolist() == [0, 1, 2]
    assert frame["source_frame_index"].tolist() == [0, 2, 4]
    assert frame["complementary_info.is_intervention"].tolist() == [0, 1, 0]
    assert np.asarray(frame["action"].iloc[1]).tolist() == (np.arange(32, 48, dtype=np.float32) + 100.0).tolist()
    assert np.isfinite(np.asarray(frame["complementary_info.teleop_action"].iloc[1])).all()
    assert episodes[0]["episode_success"] == "success"
    assert episodes[0]["dropped_hold_frames"] == 2
    assert info["hil_conversion"]["mode"] == "evo_clean"
    assert _read_frame_count(dst / "videos/chunk-000/observation.images.base/episode_000000.mp4") == 3
