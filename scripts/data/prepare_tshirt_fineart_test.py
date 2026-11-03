"""Lightweight CPU I/O/codec tests; never import a policy or run training."""

from fractions import Fraction
import hashlib
import io
import json

import av
import numpy as np
import prepare_tshirt_fineart as prep
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def video(path, frames=100, size=224, *, frozen=False):
    image = np.random.default_rng(8).integers(40, 200, (size, size, 3), dtype=np.uint8)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "30", "g": "30", "bf": "0", "threads": "1"}
        for index in range(frames):
            pixels = image if frozen else np.roll(image, index, axis=1)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def row(eid, length=60):
    value = {
        "split": "train",
        "episode_index": eid,
        "tasks": [prep.TASK],
        "length": length,
        "data/chunk_index": 0,
        "data/file_index": 0,
    }
    for camera in prep.CAMERAS:
        key = "videos/observation.images." + camera
        value.update(
            {
                key + "/chunk_index": 0,
                key + "/file_index": 0,
                key + "/from_timestamp": 0.3,
                key + "/to_timestamp": 0.3 + length / 30,
            }
        )
    return value


def test_mirror_must_match_fixed_hf_size_and_sha_before_read(tmp_path, monkeypatch):
    path = "data/chunk-000/file-000.parquet"
    plan = {"files": {path: {"size": 20, "lfs": {"oid": "a" * 64}}}}
    response = {
        "Code": 200,
        "Data": {"Files": [{"Path": path, "Type": "blob", "Size": 20, "Sha256": "b" * 64}], "TotalCount": 1},
    }
    monkeypatch.setattr(prep, "request_bytes", lambda *a, **k: (200, {}, "unused", json.dumps(response).encode()))
    with pytest.raises(prep.InfrastructureError, match="differs from pinned HF"):
        prep.MirrorInventory(tmp_path, plan).verify(path)


def test_range_cache_resumes_and_rejects_changed_block(tmp_path, monkeypatch):
    blob = bytes(range(100))
    identity = {"path": "x.mp4", "size": 100, "sha256": hashlib.sha256(blob).hexdigest()}
    calls = []

    def fetch(url, *, headers, **kwargs):
        start, end = map(int, headers["Range"].split("=")[1].split("-"))
        calls.append((start, end))
        return 206, {"Content-Range": f"bytes {start}-{end}/100"}, "signed-url-memory-only", blob[start : end + 1]

    monkeypatch.setattr(prep, "request_bytes", fetch)
    with prep.RangeReader(tmp_path, identity, block_size=8) as reader:
        reader.seek(-7, 2)
        assert reader.read(200) == blob[93:]
        reader.seek(0)
        assert reader.read() == blob
        assert len(reader.cache) <= 4
    first_count = len(calls)
    with prep.RangeReader(tmp_path, identity, block_size=8) as reader:
        assert reader.read() == blob
    assert len(calls) == first_count
    block = tmp_path / "source-blocks" / identity["sha256"] / "00000000.block"
    block.write_bytes(b"changed!")
    with pytest.raises(prep.InfrastructureError, match="Range/cache I/O"):
        prep.RangeReader(tmp_path, identity, block_size=8).read(1)
    assert block.read_bytes() == b"changed!"


def test_range_protocol_failure_is_not_structural(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "request_bytes", lambda *a, **k: (200, {}, "x", b"four"))
    with pytest.raises(prep.InfrastructureError):
        prep.RangeReader(tmp_path, {"path": "x", "size": 4, "sha256": "a" * 64}).read()
    assert not (tmp_path / "quarantine").exists()


def test_receipt_before_rename_recovers_without_overwrite(tmp_path):
    temporary = tmp_path / "output.partial"
    temporary.write_bytes(b"original")
    output = tmp_path / "output"
    item = {"path": "output", "partial_path": "output.partial", "bytes": 8, "sha256": prep.digest(temporary)}
    prep.recover_commit(tmp_path, output, tmp_path / "receipt.json", item)
    assert output.read_bytes() == b"original"
    assert not temporary.exists()
    output.write_bytes(b"modified")
    with pytest.raises(prep.InfrastructureError, match="changed"):
        prep.recover_commit(tmp_path, output, tmp_path / "receipt.json", item)


def test_long_frozen_episode_is_review_only(tmp_path):
    path = tmp_path / "long.mp4"
    video(path, frames=3605, size=64, frozen=True)
    metrics, flags = prep.audit_video(path, 0, 3605)
    assert metrics["frames"] == 3605
    assert metrics["frames"] / 30 > 120
    assert metrics["frozen_seconds"] > 1
    assert "REVIEW_static_run" in flags


def test_remux_audits_exact_interval_with_keyframe_preroll(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    video(source)
    with source.open("rb") as stream:
        offset = prep.remux_video(stream, row(1), "top", target, expected_codec="h264")
    assert offset == pytest.approx(0.3)
    metrics, _ = prep.audit_video(target, offset, 60)
    assert metrics["frames"] == 60
    assert metrics["total_decoded_with_preroll"] == 69


def pipeline(tmp_path, monkeypatch):
    root = tmp_path / "prepared"
    root.mkdir()
    source = tmp_path / "source.parquet"
    source_video = tmp_path / "source.mp4"
    video(source_video)
    values = {
        "episode_index": [1] * 60 + [2] * 60,
        "frame_index": list(range(60)) * 2,
        "timestamp": [i / 30 for i in range(60)] * 2,
        "observation.state": [[0.0] * 14] * 120,
        "action": [[0.25] * 14] * 120,
        "custom_original_language": ["preserve-me"] * 120,
    }
    pq.write_table(pa.table(values), source)
    plan = {"pilot_ids": [1], "episodes": [{"pool": "train", "source": row(1)}, {"pool": "dev", "source": row(2)}]}
    prep.save_json(root / "plan.json", plan)
    monkeypatch.setattr(prep.MirrorInventory, "verify", lambda self, p: {"path": p, "size": 1, "sha256": "a" * 64})
    calls = []

    def cached(*args):
        calls.append(1)
        return source

    monkeypatch.setattr(prep, "source_parquet", cached)
    original_remux = prep.remux_video

    def clip(reader, data, camera, target):
        with source_video.open("rb") as local:
            return original_remux(local, data, camera, target, expected_codec="h264")

    monkeypatch.setattr(prep, "remux_video", clip)

    class Reader(io.BytesIO):
        transferred = 0

        def __init__(self, *a, **kw):
            super().__init__()

    monkeypatch.setattr(prep, "RangeReader", Reader)
    return root, plan, calls


def test_limit_counts_new_episodes_and_all_original_columns_survive(tmp_path, monkeypatch):
    root, plan, calls = pipeline(tmp_path, monkeypatch)
    prep.run(root, plan, pilot_only=True, limit=1, min_free_gb=1)
    assert len(calls) == 1
    first = json.loads((root / "manifests/train.jsonl").read_text())
    assert first["status"] == "PREPARED_REVIEW_PENDING"
    table = pq.read_table(root / first["data"]["path"])
    assert table["custom_original_language"].to_pylist() == ["preserve-me"] * 60
    assert table["action"].to_pylist() == [[0.25] * 14] * 60
    prep.run(root, plan, limit=1, min_free_gb=1)
    assert prep.load(root / "completion.json")["processed"] == 2
    assert not prep.load(root / "completion.json")["training_ready"]
    before = len(calls)
    prep.run(root, plan, limit=1, min_free_gb=1)
    assert len(calls) == before


def test_transport_failure_stays_pending_and_resumes(tmp_path, monkeypatch):
    root, plan, _ = pipeline(tmp_path, monkeypatch)
    original = prep.remux_video

    def fail(*args, **kwargs):
        raise prep.InfrastructureError("connection lost")

    monkeypatch.setattr(prep, "remux_video", fail)
    with pytest.raises(prep.InfrastructureError, match="connection lost"):
        prep.run(root, plan, pilot_only=True, min_free_gb=1)
    assert not (root / "quarantine").exists()
    assert not (root / "raw/train/episode-000001/record.json").exists()
    monkeypatch.setattr(prep, "remux_video", original)
    prep.run(root, plan, pilot_only=True, min_free_gb=1)
    assert prep.load(root / "raw/train/episode-000001/record.json")["status"] == "PREPARED_REVIEW_PENDING"


def test_structural_error_quarantines_whole_episode_but_preserves_parent(tmp_path, monkeypatch):
    root, plan, _ = pipeline(tmp_path, monkeypatch)
    plan["episodes"][0]["source"]["length"] = 61
    prep.run(root, plan, pilot_only=True, min_free_gb=1)
    record = prep.load(root / "raw/train/episode-000001/record.json")
    assert record["status"] == "QUARANTINED_STRUCTURE"
    assert (root / "quarantine/train/episode-000001.json").exists()
    assert (root / "raw/train/episode-000001/source-episode.json").exists()
    assert (root / "manifests/train.jsonl").read_text() == ""


def test_pending_siblings_read_shared_source_parquet_only_once(tmp_path, monkeypatch):
    root, plan, calls = pipeline(tmp_path, monkeypatch)
    prep.run(root, plan, limit=2, min_free_gb=1)
    assert len(calls) == 1
    assert prep.load(root / "completion.json")["processed"] == 2


def test_block_receipt_recovers_interrupted_atomic_rename(tmp_path, monkeypatch):
    blob = b"saved range"
    identity = {"path": "x", "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    reader = prep.RangeReader(tmp_path, identity)
    partial = reader.directory / "interrupted.partial"
    partial.write_bytes(blob)
    prep.save_json(
        reader.directory / "00000000.json",
        {
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "partial": partial.name,
        },
    )
    monkeypatch.setattr(prep, "request_bytes", lambda *a, **k: pytest.fail("unexpected transfer"))
    assert reader.read() == blob
    assert not partial.exists()


def test_candidate_pool_overlap_is_rejected_before_output_mutation(tmp_path):
    sources = {}
    for source in prep.source_paths(row(1)).values():
        sources[source] = {"path": source, "type": "file", "size": 1, "lfs": {"size": 1, "oid": "a" * 64}}
    metadata = tmp_path / "inventory.json"
    prep.save_json(metadata, {"repo": prep.REPO, "revision": prep.HF_REVISION, "files": list(sources.values())})
    pilots = tmp_path / "pilots.json"
    prep.save_json(pilots, {"pilots": [{"source_episode_index": 1}]})
    paths = {}
    for pool in prep.POOLS:
        paths[pool] = tmp_path / (pool + ".json")
        prep.save_json(paths[pool], [row(1)] if pool != "reserve" else [])
    with pytest.raises(prep.InfrastructureError, match="pools overlap"):
        prep.make_plan(paths, metadata, pilots)
