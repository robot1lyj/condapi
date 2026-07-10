import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from scripts import build_openarm_kai0_awbc_dataset as _builder


def _write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _data_path(root: pathlib.Path, episode_index: int) -> pathlib.Path:
    return root / f"data/chunk-000/episode_{episode_index:06d}.parquet"


def _video_path(root: pathlib.Path, episode_index: int, video_key: str) -> pathlib.Path:
    return root / f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"


def _info(total_episodes: int) -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": "openarm",
        "total_episodes": total_episodes,
        "total_frames": total_episodes * 20,
        "total_tasks": 1,
        "total_videos": total_episodes * 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16], "names": None},
            "action": {"dtype": "float32", "shape": [16], "names": None},
            "observation.images.base": {"dtype": "video", "shape": [480, 640, 3], "names": None},
            "observation.images.left_wrist": {"dtype": "video", "shape": [360, 640, 3], "names": None},
            "observation.images.right_wrist": {"dtype": "video", "shape": [360, 640, 3], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _frame(episode_index: int, *, scored: bool) -> pd.DataFrame:
    length = 20
    frame = pd.DataFrame(
        {
            "observation.state": [np.zeros(16, dtype=np.float32) for _ in range(length)],
            "action": [np.ones(16, dtype=np.float32) for _ in range(length)],
            "task_index": np.zeros(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "index": np.arange(episode_index * length, (episode_index + 1) * length, dtype=np.int64),
        }
    )
    if scored:
        frame["relative_advantage"] = np.tile(np.arange(10, dtype=np.float32) / 9.0, 2)
        frame["absolute_value"] = np.concatenate(
            [np.full(10, 0.25, dtype=np.float32), np.full(10, 0.75, dtype=np.float32)]
        )
        frame["absolute_advantage"] = np.zeros(length, dtype=np.float32)
    return frame


def _write_dataset(root: pathlib.Path, source_episode_ids: list[int], *, scored: bool, kind: str) -> None:
    info = _info(len(source_episode_ids))
    _write_json(root / "meta/info.json", info)
    rows = []
    for local_episode, source_episode in enumerate(source_episode_ids):
        parquet = _data_path(root, local_episode)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        _frame(local_episode, scored=scored).to_parquet(parquet, index=False)
        for video_key in _builder.VIDEO_KEYS:
            video = _video_path(root, local_episode, video_key)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
        row = {
            "episode_index": local_episode,
            "source_episode_index": source_episode,
            "length": 20,
            "tasks": ["Fold the T-shirt properly"],
        }
        if kind in {"time", "mirror"}:
            row.update(
                {
                    "augmentation_type": kind,
                    "source_frame_stride": 1,
                    "source_frame_offset": 0,
                    "mirror": kind == "mirror",
                }
            )
        rows.append(row)
    _write_jsonl(root / "meta/episodes.jsonl", rows)
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "Fold the T-shirt properly"}])


def _write_tda_dataset(root: pathlib.Path) -> None:
    info = _info(2)
    _write_json(root / "meta/info.json", info)
    rows = []
    for episode_index, (source_episode, kind) in enumerate(((0, "time"), (1, "mirror"))):
        parquet = _data_path(root, episode_index)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        _frame(episode_index, scored=False).to_parquet(parquet, index=False)
        for video_key in _builder.VIDEO_KEYS:
            video = _video_path(root, episode_index, video_key)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
        rows.append(
            {
                "episode_index": episode_index,
                "source_episode_index": source_episode,
                "length": 20,
                "tasks": ["Fold the T-shirt properly"],
                "augmentation_type": kind,
                "source_frame_stride": 1,
                "source_frame_offset": 0,
                "mirror": kind == "mirror",
            }
        )
    _write_jsonl(root / "meta/episodes.jsonl", rows)
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "Fold the T-shirt properly"}])


def test_builds_binary_awbc_dataset_with_source_audit(tmp_path: pathlib.Path) -> None:
    hq = tmp_path / "hq"
    site = tmp_path / "site"
    tda = tmp_path / "tda"
    destination = tmp_path / "awbc"
    _write_dataset(hq, [0, 1], scored=True, kind="HQ")
    _write_dataset(site, [0], scored=True, kind="Site")
    _write_tda_dataset(tda)

    report = _builder.build_kai0_awbc_dataset(
        [hq],
        [site],
        tda,
        destination,
        site_repeat=2,
        tda_time_count=0,
        tda_mirror_count=0,
        positive_ratio=0.30,
        relative_interval=5,
        overwrite=False,
        expected_hq_episodes=2,
        site_train_source_end=1,
        excluded_site_source_episodes=(),
    )

    assert report["total_episodes"] == 4
    assert report["total_frames"] == 80
    assert report["materialized_episode_counts"] == {"HQ": 2, "Site": 2, "TDA": 0}
    assert report["overall_unique_label_counts"]["0"]["positive_ratio"] == 0.3
    assert report["overall_unique_label_counts"]["1"]["positive_ratio"] == 0.3
    tasks = [json.loads(line) for line in (destination / "meta/tasks.jsonl").read_text().splitlines()]
    assert tasks == [
        {"task_index": 0, "task": "Fold the T-shirt properly, Advantage: negative"},
        {"task_index": 1, "task": "Fold the T-shirt properly, Advantage: positive"},
    ]
    first = pd.read_parquet(destination / "data/chunk-000/episode_000000.parquet")
    assert set(first["task_index"].unique()) == {0, 1}
    assert set(first["stage_id_pred"].unique()) == {0, 1}
    assert (destination / "kai0_awbc_build_report.json").exists()
    assert not list(tmp_path.glob(".awbc.building-*"))


def test_rejects_incomplete_hq_scores_before_creating_destination(tmp_path: pathlib.Path) -> None:
    hq = tmp_path / "hq"
    site = tmp_path / "site"
    tda = tmp_path / "tda"
    destination = tmp_path / "awbc"
    _write_dataset(hq, [0], scored=True, kind="HQ")
    _write_dataset(site, [0], scored=True, kind="Site")
    _write_tda_dataset(tda)

    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        _builder.build_kai0_awbc_dataset(
            [hq],
            [site],
            tda,
            destination,
            site_repeat=1,
            tda_time_count=1,
            tda_mirror_count=1,
            positive_ratio=0.30,
            relative_interval=5,
            overwrite=False,
            expected_hq_episodes=2,
            site_train_source_end=1,
            excluded_site_source_episodes=(),
        )
    assert not destination.exists()
