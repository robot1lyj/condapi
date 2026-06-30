import json

import numpy as np
import pandas as pd

from scripts import subset_lerobot_v21


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _episode_df(episode_index, length, start_index):
    return pd.DataFrame(
        {
            "action": [np.full(16, episode_index, dtype=np.float32) for _ in range(length)],
            "observation.state": [np.full(16, episode_index, dtype=np.float32) for _ in range(length)],
            "timestamp": np.arange(length, dtype=np.float32) / 30.0,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(start_index, start_index + length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
        }
    )


def test_create_subset_reindexes_parquets_and_meta(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 3,
        "total_frames": 18,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [16], "names": None},
            "observation.state": {"dtype": "float32", "shape": [16], "names": None},
            "observation.images.base": {"dtype": "video", "shape": [4, 4, 3], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    _write_json(src / "meta/info.json", info)
    _write_jsonl(src / "meta/tasks.jsonl", [{"task_index": 0, "task": "task"}])
    _write_jsonl(
        src / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["task"], "length": 5},
            {"episode_index": 1, "tasks": ["task"], "length": 6},
            {"episode_index": 2, "tasks": ["task"], "length": 7},
        ],
    )
    start = 0
    for episode_index, length in enumerate([5, 6, 7]):
        parquet = src / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        _episode_df(episode_index, length, start).to_parquet(parquet, index=False)
        video = src / f"videos/chunk-000/observation.images.base/episode_{episode_index:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{episode_index}".encode())
        start += length

    report = subset_lerobot_v21.create_subset(
        src,
        dst,
        [1, 2],
        copy_mode="copy",
        overwrite=False,
        val_count=1,
    )
    first = pd.read_parquet(dst / "data/chunk-000/episode_000000.parquet")
    second = pd.read_parquet(dst / "data/chunk-000/episode_000001.parquet")
    out_info = json.loads((dst / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (dst / "meta/episodes.jsonl").read_text().splitlines()]

    assert report["total_episodes"] == 2
    assert out_info["total_episodes"] == 2
    assert out_info["total_frames"] == 13
    assert out_info["splits"] == {"train": "0:1", "val": "1:2"}
    assert episodes[0]["episode_index"] == 0
    assert episodes[1]["episode_index"] == 1
    assert first["episode_index"].unique().tolist() == [0]
    assert second["episode_index"].unique().tolist() == [1]
    assert first["index"].iloc[0] == 0
    assert second["index"].iloc[0] == 6
    assert (dst / "videos/chunk-000/observation.images.base/episode_000001.mp4").read_bytes() == b"video-2"
