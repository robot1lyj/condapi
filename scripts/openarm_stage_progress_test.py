import json

import numpy as np
import pandas as pd

from scripts import openarm_stage_progress as stage_progress


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_stage_arrays_two_stage_boundaries():
    boundaries = [
        {"stage_id": 0, "name": "flattening", "start_frame": 0, "end_frame": 4},
        {"stage_id": 1, "name": "folding", "start_frame": 5, "end_frame": 9},
    ]
    progress, stage_ids = stage_progress.build_stage_arrays(10, boundaries)

    assert progress.dtype == np.float32
    assert stage_ids.dtype == np.int64
    assert np.all(np.diff(progress) >= -1e-6)
    assert progress[0] == 0.0
    assert progress[4] == 0.5
    assert progress[5] == 0.5
    assert progress[-1] == 1.0
    assert np.all(stage_ids[:5] == 0)
    assert np.all(stage_ids[5:] == 1)


def test_apply_annotations_writes_stage_progress(tmp_path):
    dataset = tmp_path / "dataset"
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_frames": 10,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [16], "names": None},
            "observation.state": {"dtype": "float32", "shape": [16], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    _write_json(dataset / "meta/info.json", info)
    _write_jsonl(dataset / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": ["task"], "length": 10}])

    episode_frame = pd.DataFrame(
        {
            "action": [np.zeros(16, dtype=np.float32) for _ in range(10)],
            "observation.state": [np.zeros(16, dtype=np.float32) for _ in range(10)],
            "timestamp": np.arange(10, dtype=np.float32) / 30.0,
            "frame_index": np.arange(10, dtype=np.int64),
            "episode_index": np.zeros(10, dtype=np.int64),
            "index": np.arange(10, dtype=np.int64),
            "task_index": np.zeros(10, dtype=np.int64),
        }
    )
    parquet_path = dataset / "data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    episode_frame.to_parquet(parquet_path, index=False)
    annotations = dataset / "annotations/openarm_stage_v1.jsonl"
    _write_jsonl(
        annotations,
        [
            {
                "episode_index": 0,
                "quality": "success",
                "events": [
                    {"name": "episode_start", "frame": 0},
                    {"name": "flatten_done", "frame": 4},
                    {"name": "fold_start", "frame": 5},
                    {"name": "episode_end", "frame": 9},
                ],
            }
        ],
    )

    report = stage_progress.apply_annotations(dataset, annotations, dry_run=False, quality_filter={"success"})
    out = pd.read_parquet(parquet_path)
    updated_info = json.loads((dataset / "meta/info.json").read_text())

    assert report["processed_count"] == 1
    assert "stage_progress_gt" in out.columns
    assert "stage_id" in out.columns
    assert out["stage_progress_gt"].iloc[4] == 0.5
    assert out["stage_progress_gt"].iloc[-1] == 1.0
    assert updated_info["features"]["stage_progress_gt"]["dtype"] == "float32"
