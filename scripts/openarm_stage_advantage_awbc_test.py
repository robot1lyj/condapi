import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import openarm_stage_advantage_awbc as awbc


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _make_source_dataset(root):
    features = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for video_key in awbc.OPENARM_VIDEO_KEYS:
        features[video_key] = {"dtype": "video", "shape": [8, 10, 3], "names": ["height", "width", "channels"]}

    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_frames": 2,
        "total_videos": 3,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    _write_json(root / "meta/info.json", info)
    _write_jsonl(root / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": ["old task"], "length": 2}])

    frame = pd.DataFrame(
        {
            "observation.state": [np.zeros(16, dtype=np.float32) for _ in range(2)],
            "task_index": np.zeros(2, dtype=np.int64),
            "episode_index": np.zeros(2, dtype=np.int64),
            "frame_index": np.arange(2, dtype=np.int64),
            "index": np.arange(2, dtype=np.int64),
        }
    )
    parquet_path = root / "data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    frame.to_parquet(parquet_path, index=False)

    for video_key in awbc.OPENARM_VIDEO_KEYS:
        video_path = root / f"videos/chunk-000/{video_key}/episode_000000.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()


def test_score_only_writes_raw_scores_without_discretizing(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "scores"
    _make_source_dataset(source)

    monkeypatch.setattr(awbc, "_load_model", lambda *_args, **_kwargs: (SimpleNamespace(), object()))
    monkeypatch.setattr(
        awbc,
        "_predict_episode",
        lambda **_kwargs: {
            "relative_advantage": np.asarray([0.1, 0.2], dtype=np.float32),
            "absolute_value": np.asarray([0.0, 0.4], dtype=np.float32),
            "absolute_advantage": np.asarray([0.4, 0.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        awbc,
        "_assign_awbc_labels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("score-only must not discretize")),
    )

    args = SimpleNamespace(
        src=source,
        dst=destination,
        checkpoint=tmp_path / "checkpoint",
        config_name="ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD",
        episodes="0:1",
        task="Fold the T-shirt properly",
        relative_interval=50,
        batch_size=32,
        samples_per_batch=1,
        bad_percentile=20.0,
        positive_percentile=70.0,
        copy_mode="hardlink",
        device="cpu",
        seed=42,
        score_only=True,
        overwrite=False,
        resume=False,
        dry_run=False,
    )

    report = awbc.build_awbc_dataset(args)

    output = pd.read_parquet(destination / "data/chunk-000/episode_000000.parquet")
    info = json.loads((destination / "meta/info.json").read_text())
    tasks = [json.loads(line) for line in (destination / "meta/tasks.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (destination / "meta/episodes.jsonl").read_text().splitlines()]

    assert report["score_only"] is True
    assert report["discretize"] is None
    assert info["total_tasks"] == 1
    assert tasks == [{"task_index": 0, "task": "Fold the T-shirt properly"}]
    assert episodes[0]["source_episode_index"] == 0
    assert episodes[0]["tasks"] == ["Fold the T-shirt properly"]
    assert output["task_index"].tolist() == [0, 0]
    np.testing.assert_allclose(output["relative_advantage"].to_numpy(), [0.1, 0.2])
    assert not (destination / "awbc_discretize_report.json").exists()

    args.resume = True
    monkeypatch.setattr(
        awbc,
        "_predict_episode",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("valid completed episode must be skipped")),
    )
    resumed_report = awbc.build_awbc_dataset(args)
    assert resumed_report["resumed_episodes"] == 1
    assert resumed_report["newly_scored_episodes"] == 0
