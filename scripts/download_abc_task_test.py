"""CPU file/codec checks only; no model or training loop."""

from fractions import Fraction
import io
import json
from pathlib import Path
import urllib.request

import av
from download_abc_task import CAMERAS
from download_abc_task import POLICY
from download_abc_task import RangeReader
from download_abc_task import audit_video
from download_abc_task import clip_video
from download_abc_task import run
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def make_video(path, *, frozen=False):
    rng = np.random.default_rng(12)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "g": "30", "bf": "0"}
        image = rng.integers(30, 220, (64, 64, 3), dtype=np.uint8)
        for i in range(100):
            pixels = image if frozen else np.roll(image, i, axis=1)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def test_ranges_seek_eof_and_bounded_cache(monkeypatch):
    blob = bytes(range(100))

    class Response(io.BytesIO):
        status = 206

        def geturl(self):
            return "https://example.test/file"

    def open_range(request, timeout):
        start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
        response = Response(blob[start : end + 1])
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(blob)}"}
        return response

    monkeypatch.setattr(urllib.request, "urlopen", open_range)
    reader = RangeReader("https://example.test/file", 100, block_size=8)
    assert reader.read(3) == blob[:3]
    assert reader.seek(-5, 2) == 95
    assert reader.read(1000) == blob[95:]
    assert reader.read(10) == b""
    reader.seek(0)
    assert reader.read() == blob
    assert len(reader.cache) <= 4
    with pytest.raises(ValueError, match="invalid seek"):
        reader.seek(-1)


def test_ignored_range_is_infrastructure_failure(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = None

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response(b"full file"))
    monkeypatch.setattr("download_abc_task.time.sleep", lambda _: None)
    with pytest.raises(OSError, match="not classified as bad data"):
        RangeReader("https://example.test/file", 100).read(1)


def test_remux_preserves_pixels_with_keyframe_preroll(tmp_path):
    source, target = tmp_path / "source.mp4", tmp_path / "target.mp4"
    make_video(source)
    row = {
        "length": 60,
        "videos/observation.images.top/from_timestamp": 0.3,
        "videos/observation.images.top/to_timestamp": 2.3,
    }
    with source.open("rb") as reader:
        offset = clip_video(reader, row, "top", target)
    assert offset == pytest.approx(0.3)
    metrics, flags = audit_video(target, offset, 60)
    assert metrics["frames"] == 60
    assert flags == []
    with av.open(str(source)) as original, av.open(str(target)) as extracted:
        expected = [f.to_ndarray(format="rgb24") for f in original.decode(video=0)][9:69]
        actual = [f.to_ndarray(format="rgb24") for f in extracted.decode(video=0)][9:69]
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True))


def test_continuous_freeze_quarantined(tmp_path):
    path = tmp_path / "frozen.mp4"
    make_video(path, frozen=True)
    _, flags = audit_video(path, 0, 90)
    assert "video_frozen_run" in flags


def test_pipeline_keeps_only_target_and_stops_on_io(tmp_path, monkeypatch):
    source, video = tmp_path / "source.parquet", tmp_path / "source.mp4"
    make_video(video)
    table = pa.table(
        {
            "episode_index": [4] * 60 + [5] * 2,
            "observation.state": [[0.0] * 14] * 62,
            "action": [[0.0] * 14] * 62,
            "frame_index": [*range(60), 0, 1],
            "timestamp": [*(np.arange(60) / 30), 0.0, 1 / 30],
        }
    )
    pq.write_table(table, source)
    path = "data/chunk-000/file-000.parquet"
    row = {"episode_index": 4, "length": 60, "data/chunk_index": 0, "data/file_index": 0}
    files = {path: {"Path": str(source)}}
    for camera in CAMERAS:
        key = f"videos/observation.images.{camera}"
        row.update(
            {key + "/chunk_index": 0, key + "/file_index": 0, key + "/from_timestamp": 0.3, key + "/to_timestamp": 2.3}
        )
        files[f"{key}/chunk-000/file-000.mp4"] = {"Path": str(video)}
    plan = {
        "task": "fold the paper box",
        "policy": POLICY,
        "splits": {"train": {"source_repo": "test", "mirror_revision": "fixed", "episodes": [row], "files": files}},
    }
    root = tmp_path / "download"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan))

    class Reader:
        transferred = 123

        def __init__(self, path):
            self.stream = Path(path).open("rb")  # noqa: SIM115 -- closed by __exit__

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def __getattr__(self, key):
            return getattr(self.stream, key)

    monkeypatch.setattr("download_abc_task.remote", lambda repo, revision, record: Reader(record["Path"]))
    monkeypatch.setattr("download_abc_task.shutil.disk_usage", lambda _: type("Disk", (), {"free": 10**15})())
    run(root, plan)
    published = [json.loads(line) for line in (root / "manifests/train.jsonl").read_text().splitlines()]
    assert [r["source_episode_index"] for r in published] == [4]
    assert pq.read_table(root / "train" / path)["episode_index"].to_pylist() == [4] * 60
    assert published[0]["videos"]["top"]["offset_s"] == pytest.approx(0.3)
    # Completed members resume without a new transfer.
    monkeypatch.setattr("download_abc_task.remote", lambda *args: pytest.fail("unexpected transfer"))
    run(root, plan)
    assert len((root / "audit.jsonl").read_text().splitlines()) == 1
