import hashlib
import os
import time

import av
import numpy as np
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
