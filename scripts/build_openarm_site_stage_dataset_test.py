import json

import numpy as np
import pandas as pd

from scripts import build_openarm_site_stage_dataset as builder


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _make_source(root):
    video_keys = (
        "observation.images.base",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    features = {
        "observation.state": {"dtype": "float32", "shape": [16]},
        "action": {"dtype": "float32", "shape": [16]},
        **{key: {"dtype": "video", "shape": [4, 6, 3]} for key in video_keys},
    }
    info = {
        "total_episodes": 3,
        "total_frames": 12,
        "total_videos": 9,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    _write_json(root / "meta/info.json", info)
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [{"episode_index": index, "length": 4, "tasks": ["old"]} for index in range(3)],
    )
    for episode_index in range(3):
        frame = pd.DataFrame(
            {
                "observation.state": [np.zeros(16, dtype=np.float32) for _ in range(4)],
                "action": [np.zeros(16, dtype=np.float32) for _ in range(4)],
                "episode_index": np.full(4, episode_index, dtype=np.int64),
                "frame_index": np.arange(4, dtype=np.int64),
                "index": np.arange(episode_index * 4, episode_index * 4 + 4, dtype=np.int64),
                "task_index": np.zeros(4, dtype=np.int64),
            }
        )
        parquet = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet, index=False)
        for key in video_keys:
            video = root / f"videos/chunk-000/{key}/episode_{episode_index:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()


def test_build_site_stage_dataset_filters_failure_and_keeps_validation(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "stage"
    _make_source(source)
    annotations = source / "annotations/openarm_stage_v1.jsonl"
    _write_jsonl(
        annotations,
        [
            {"episode_index": 0, "quality": "success", "flatten_done": 1},
            {"episode_index": 1, "quality": "failure", "flatten_done": 1},
            {"episode_index": 2, "quality": "success", "flatten_done": 1},
        ],
    )

    report = builder.build_site_stage_dataset(
        source,
        annotations,
        destination,
        validation_start=2,
        expected_train=1,
        expected_validation=1,
        overwrite=False,
    )

    info = json.loads((destination / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (destination / "meta/episodes.jsonl").read_text().splitlines()]
    episode_stats = [json.loads(line) for line in (destination / "meta/episodes_stats.jsonl").read_text().splitlines()]
    train_frame = pd.read_parquet(destination / "data/chunk-000/episode_000000.parquet")
    val_frame = pd.read_parquet(destination / "data/chunk-000/episode_000001.parquet")
    assert report["source_train_episode_indices"] == [0]
    assert report["source_validation_episode_indices"] == [2]
    assert info["splits"] == {"train": "0:1", "validation": "1:2"}
    assert [row["source_episode_index"] for row in episodes] == [0, 2]
    assert [row["episode_index"] for row in episode_stats] == [0, 1]
    assert "stage_progress_gt" in episode_stats[0]["stats"]
    assert train_frame["stage_progress_gt"].tolist() == [0.0, 0.5, 0.5, 1.0]
    assert val_frame["stage_id"].tolist() == [0, 0, 1, 1]
    assert "stage_progress_gt" not in pd.read_parquet(source / "data/chunk-000/episode_000000.parquet").columns
