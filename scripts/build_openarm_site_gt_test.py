import json

import numpy as np
import pandas as pd

from scripts import build_openarm_site_gt as site_gt
from scripts import build_openarm_site_gt_report as site_gt_report


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _make_site_dataset(root):
    features = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": None},
        "action": {"dtype": "float32", "shape": [16], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for video_key in site_gt.VIDEO_KEYS:
        features[video_key] = {"dtype": "video", "shape": [8, 10, 3], "names": None}
    info = {
        "codebase_version": "v2.1",
        "repo_id": "site",
        "total_episodes": 1,
        "total_frames": 10,
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
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "Fold the T-shirt properly"}])
    _write_jsonl(root / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": ["task"], "length": 10}])
    frame = pd.DataFrame(
        {
            "observation.state": [np.zeros(16, dtype=np.float32) for _ in range(10)],
            "action": [np.zeros(16, dtype=np.float32) for _ in range(10)],
            "task_index": np.zeros(10, dtype=np.int64),
            "episode_index": np.zeros(10, dtype=np.int64),
            "frame_index": np.arange(10, dtype=np.int64),
            "index": np.arange(10, dtype=np.int64),
        }
    )
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    frame.to_parquet(parquet, index=False)
    for video_key in site_gt.VIDEO_KEYS:
        video = root / f"videos/chunk-000/{video_key}/episode_000000.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
    _write_jsonl(
        root / "annotations/openarm_stage_v1.jsonl",
        [
            {
                "episode_index": 0,
                "quality": "success",
                "events": [
                    {"name": "episode_start", "frame": 0},
                    {"name": "flatten_done", "frame": 4},
                    {"name": "episode_end", "frame": 9},
                ],
            }
        ],
    )


def test_build_gt_advantage_normalizes_final_window():
    progress = np.arange(6, dtype=np.float32) / 5.0
    advantage = site_gt.build_gt_advantage(progress, relative_interval=2)

    np.testing.assert_allclose(advantage[:-1], 0.4)
    assert advantage[-1] == 0.0


def test_build_site_gt_dataset_keeps_source_unchanged(tmp_path):
    source = tmp_path / "site"
    destination = tmp_path / "site_gt"
    _make_site_dataset(source)

    report = site_gt.build_site_gt_dataset(
        source,
        destination,
        annotations_path=None,
        relative_interval=2,
        copy_mode="hardlink",
        overwrite=False,
        dry_run=False,
    )

    source_frame = pd.read_parquet(source / "data/chunk-000/episode_000000.parquet")
    output = pd.read_parquet(destination / "data/chunk-000/episode_000000.parquet")
    info = json.loads((destination / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (destination / "meta/episodes.jsonl").read_text().splitlines()]

    assert "stage_progress_gt" not in source_frame.columns
    assert set(site_gt.ADVANTAGE_FEATURES).issubset(output.columns)
    np.testing.assert_allclose(output["absolute_value"], output["stage_progress_gt"])
    np.testing.assert_allclose(output["relative_advantage"], output["advantage_gt"])
    np.testing.assert_allclose(output["absolute_advantage"], output["advantage_gt"])
    assert info["repo_id"] == "site_gt"
    assert info["site_gt"]["hq_stage_used"] is False
    assert episodes[0]["advantage_source"] == "site_gt"
    assert episodes[0]["annotation_quality"] == "success"
    assert episodes[0]["eligible_for_k_data"] is True
    assert episodes[0]["flatten_done_frame"] == 4
    assert report["advantage_source"] == "site_gt"
    assert (destination / "videos/chunk-000/observation.images.base/episode_000000.mp4").stat().st_ino == (
        source / "videos/chunk-000/observation.images.base/episode_000000.mp4"
    ).stat().st_ino

    index_path = site_gt_report.build_html_report(
        destination,
        destination / "site_gt_report",
        overwrite=False,
    )
    payload = json.loads((destination / "site_gt_report/data/episode_000000.json").read_text())
    assert index_path.exists()
    assert "Site-GT Stage Advantage Review" in index_path.read_text()
    assert payload["flatten_done_frame"] == 4
    assert payload["videos"]["base"] == "../videos/chunk-000/observation.images.base/episode_000000.mp4"
