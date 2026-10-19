import hashlib
import json
import os
import sys
import time

import av
import numpy as np
import prepare_yam_clean_subset as cleaner
from prepare_yam_clean_subset import audit_episode
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def source_fixture(root):
    episode = {"source_episode_index": 9, "length": 2, "fps": 30, "task": "earbuds", "data": {"file_index": 0}}
    data = root / "train/data/source-file-000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [9, 9],
                "frame_index": [0, 1],
                "timestamp": [0.0, 1 / 30],
                "observation.state": np.zeros((2, 14)).tolist(),
                "action": np.zeros((2, 14)).tolist(),
            }
        ),
        data,
    )
    for camera in ("top", "left_wrist", "right_wrist"):
        video = root / "train/videos" / camera / "episode-000009.mp4"
        video.parent.mkdir(parents=True)
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for _ in range(1 if camera == "left_wrist" else 2):
                frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, (time.time() - 600, time.time() - 600))
    return episode


def test_short_video_excludes_whole_episode_without_touching_originals(tmp_path):
    episode = source_fixture(tmp_path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    result = audit_episode((str(tmp_path), "train", episode))
    assert result["status"] == "excluded"
    assert result["source_episode_index"] == 9
    assert len(result["errors"]) == 1
    assert result["errors"][0]["error"] == "video_frame_count:1!=2"
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())


def test_missing_file_stops_audit_instead_of_excluding_episode(tmp_path):
    episode = {"source_episode_index": 9, "length": 2, "fps": 30, "task": "earbuds", "data": {"file_index": 0}}
    with pytest.raises(FileNotFoundError):
        audit_episode((str(tmp_path), "train", episode))


def test_new_selection_excludes_registered_train_and_val_before_decoding(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "manifests").mkdir(parents=True)
    train_bad = {
        "source_repo": "lerobot/abc_130k_v3_train",
        "source_revision": "68651e4929d9fb00f798937b2d62617cab5c771d",
        "source_episode_index": 101616,
        "length": 5400000,
        "fps": 30,
    }
    val_bad = {
        **train_bad,
        "source_repo": "lerobot/abc_130k_v3_val",
        "source_revision": "46ca817c39d3df08115a75c31b3cd3867e3d875a",
        "source_episode_index": 455,
        "length": 100,
    }
    for split, rows in {
        "train": [train_bad, {**train_bad, "source_episode_index": 101617}],
        "val": [val_bad, {**val_bad, "source_episode_index": 456}],
    }.items():
        (source / "manifests" / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    output = tmp_path / "filtered"
    observed = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, tasks, **kwargs):
            for _, split, row in tasks:
                idx = row["source_episode_index"]
                assert idx not in (101616, 455), "Registered exclusions must never enter decoding"
                observed.append((split, idx))
                yield {"split": split, "source_episode_index": idx, "length": row["length"], "status": "passed"}

    monkeypatch.setattr(cleaner, "ProcessPoolExecutor", Pool)
    monkeypatch.setenv("SLURM_JOB_ID", "metadata-test")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        sys,
        "argv",
        ["clean", "--source", str(source), "--output", str(output), "--contract", str(contract), "--workers", "1"],
    )
    cleaner.main()
    selection = json.loads((output / "selection.json").read_text())
    assert set(observed) == {("train", 101617), ("val", 456)}
    assert selection["excluded_source_episode_ids"] == [101616]
    assert selection["excluded_val_source_episode_ids"] == [455]
    assert selection["val_episodes"] == 1
    assert selection["known_qc_exclusions"]["train"][0]["source_episode_index"] == 101616
    assert selection["known_qc_exclusions"]["val"][0]["source_episode_index"] == 455
