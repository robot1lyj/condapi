import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import av
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training import config as training_config
from openpi.training import data_loader
from scripts.convert_yam_subset import CAMERA_KEYS
from scripts.convert_yam_subset import JOINT_NAMES
from scripts.convert_yam_subset import convert
from scripts.convert_yam_subset import digest
from scripts.yam_conversion_resume import conversion_lock


@pytest.fixture
def subset(tmp_path):
    root = tmp_path / "raw"
    (root / "manifests").mkdir(parents=True)
    for split in ("train", "val"):
        episode = {
            "source_episode_index": 95,
            "source_repo": f"fixture/{split}",
            "source_revision": "fixture-v1",
            "length": 3,
            "fps": 30,
            "task": "sort the legos into containers by color",
            "data": {"file_index": 0},
        }
        (root / "manifests" / f"{split}.jsonl").write_text(json.dumps(episode) + "\n")
        data = root / split / "data"
        data.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": [95] * 3,
                    "frame_index": [0, 1, 2],
                    "timestamp": [0, 1 / 30, 2 / 30],
                    "observation.state": np.zeros((3, 14), dtype=np.float32).tolist(),
                    "action": np.ones((3, 14), dtype=np.float32).tolist(),
                }
            ),
            data / "source-file-000.parquet",
        )
        for camera in CAMERA_KEYS:
            path = root / split / "videos" / camera / "episode-000095.mp4"
            path.parent.mkdir(parents=True)
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width = stream.height = 32
                stream.pix_fmt = "yuv420p"
                for i in range(3):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 50 + i, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
    return root


@pytest.fixture
def contract():
    return {
        "state_action_names": JOINT_NAMES,
        "joint_unit": "synthetic_rad",
        "gripper_unit": "synthetic_normalized",
        "action_mode": "absolute",
        "evidence": "synthetic fixture only; not a real YAM unit assertion",
    }


@pytest.mark.parametrize("video_mode", ["reencode", "copy"])
def test_roundtrip_preserves_values_and_split(subset, contract, tmp_path, video_mode):
    before = {p: digest(p) for p in subset.rglob("*") if p.is_file()}
    for split in ("train", "val"):
        output = convert(subset, tmp_path / f"clean_{split}", split, contract, min_age=0, video_mode=video_mode)
        data = LeRobotDataset(
            f"local/clean_{split}", root=output, video_backend="pyav", delta_timestamps={"action": [0, 1 / 30, 2 / 30]}
        )
        assert len(data) == 3
        assert data.num_episodes == 1
        sample = data[2]
        np.testing.assert_array_equal(sample["action"], np.ones((3, 14)))
        np.testing.assert_array_equal(sample["observation.state"], np.zeros(14))
        assert all(sample[key].shape == (3, 32, 32) for key in CAMERA_KEYS.values())
        provenance = json.loads((output / "conversion_manifest.json").read_text())
        assert provenance["episodes"][0]["source_repo"] == f"fixture/{split}"
        assert provenance["episodes"][0]["source_episode_index"] == 95
        assert provenance["training_verified"] is False
        if video_mode == "copy":
            for camera, key in CAMERA_KEYS.items():
                original = subset / split / "videos" / camera / "episode-000095.mp4"
                copied = output / "videos" / key / "chunk-000" / "file-000.mp4"
                assert digest(original) == digest(copied)
                assert not copied.is_symlink()
                assert original.stat().st_ino != copied.stat().st_ino
        cfg = training_config.get_config("pi05_yam_lora")
        loader = data_loader.create_torch_dataset(
            training_config.DataConfig(
                repo_id=str(output),
                prompt_from_task=True,
                action_sequence_keys=("action",),
                lerobot_video_backend="pyav",
            ),
            50,
            cfg.model,
        )
        item = loader[2]
        assert item["prompt"] == "sort the legos into containers by color"
        assert item["action"].shape == (50, 14)
    assert all(digest(p) == value for p, value in before.items())


def test_refuses_unknown_contract_or_missing_upload(subset, contract, tmp_path):
    with pytest.raises(ValueError, match="confirm"):
        convert(subset, tmp_path / "bad", "train", {}, min_age=0)
    (subset / "train/videos/top/episode-000095.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        convert(subset, tmp_path / "bad", "train", contract, min_age=0)
    assert not (tmp_path / "bad.incomplete").exists()


def test_refuses_source_output_and_missing_episode(subset, contract, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        convert(subset, subset / "output", "train", contract, min_age=0)
    with pytest.raises(ValueError, match="subset"):
        convert(subset, tmp_path / "bad", "train", contract, episode_ids=[999], min_age=0)


@pytest.mark.parametrize("video_mode", ["reencode", "copy"])
def test_changed_source_never_published(subset, contract, tmp_path, monkeypatch, video_mode):
    writer_class = LeRobotDatasetMetadata if video_mode == "copy" else LeRobotDataset
    original_save = writer_class.save_episode

    def changed_save(self, *args, **kwargs):
        result = original_save(self, *args, **kwargs)
        path = subset / "manifests/train.jsonl"
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(writer_class, "save_episode", changed_save)
    with pytest.raises(ValueError, match="Source changed"):
        convert(subset, tmp_path / "changed", "train", contract, min_age=0, video_mode=video_mode)
    assert not (tmp_path / "changed").exists()
    assert (tmp_path / "changed.incomplete").exists()


def test_copy_mode_multiple_episode_offsets(subset, contract, tmp_path):
    manifest = subset / "manifests/train.jsonl"
    first = json.loads(manifest.read_text())
    second = dict(first, source_episode_index=96, task="sort another color")
    manifest.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    parquet = subset / "train/data/source-file-000.parquet"
    rows = pq.read_table(parquet).to_pydict()
    expanded = {key: value + value for key, value in rows.items()}
    expanded["episode_index"] = [95] * 3 + [96] * 3
    expanded["action"] = [[1.0] * 14] * 3 + [[0.5] * 14] * 3
    pq.write_table(pa.table(expanded), parquet)
    for camera in CAMERA_KEYS:
        source = subset / "train/videos" / camera / "episode-000095.mp4"
        shutil.copyfile(source, source.with_name("episode-000096.mp4"))
    output = convert(subset, tmp_path / "multi", "train", contract, min_age=0, video_mode="copy")
    data = LeRobotDataset("local/multi", root=output, video_backend="pyav", delta_timestamps={"action": [0, 1 / 30]})
    assert len(data) == 6
    assert data.num_episodes == 2
    np.testing.assert_array_equal(data[2]["action"], np.ones((2, 14)))
    np.testing.assert_array_equal(data[3]["action"], np.full((2, 14), 0.5))
    assert data[3]["task"] == "sort another color"
    assert (output / "videos/observation.images.top_rgb/chunk-000/file-001.mp4").is_file()


@pytest.mark.parametrize("damage_completed_video", [False, True])
def test_resume_after_interruption(subset, contract, tmp_path, monkeypatch, damage_completed_video):
    from scripts import convert_yam_subset as converter  # noqa: PLC0415

    original = converter.validate_video
    calls = []

    def interrupt(path, length, fps):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(path, length, fps)

    monkeypatch.setattr(converter, "validate_video", interrupt)
    output = tmp_path / "resumable"
    with pytest.raises(RuntimeError, match="simulated"):
        convert(subset, output, "train", contract, min_age=0, video_mode="copy")
    work = tmp_path / "resumable.incomplete"
    copied = work / "videos/observation.images.top_rgb/chunk-000/file-000.mp4"
    before = copied.stat().st_mtime_ns
    if damage_completed_video:
        copied.write_bytes(b"damaged derived copy")
    calls.clear()

    def count(path, length, fps):
        calls.append(path)
        return original(path, length, fps)

    monkeypatch.setattr(converter, "validate_video", count)
    convert(subset, output, "train", contract, min_age=0, video_mode="copy", resume=True)
    assert len(calls) == (3 if damage_completed_video else 2)
    final_video = output / copied.relative_to(work)
    if not damage_completed_video:
        assert final_video.stat().st_mtime_ns == before
    else:
        assert list(final_video.parent.glob("*.interrupted-*"))
    data = LeRobotDataset("local/resumable", root=output, video_backend="pyav")
    assert len(data) == 3
    calls.clear()
    assert convert(subset, output, "train", contract, min_age=0, video_mode="copy", resume=True) == output
    assert calls == []
    parquet = output / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(parquet).to_pydict()
    rows["action"][0][0] = 0.123
    pq.write_table(pa.table(rows), parquet)
    with pytest.raises(ValueError, match="Completed parquet changed"):
        convert(subset, output, "train", contract, min_age=0, video_mode="copy", resume=True)


def test_resume_rejects_changed_configuration_and_unknown_work(subset, contract, tmp_path, monkeypatch):
    from scripts import convert_yam_subset as converter  # noqa: PLC0415

    def interrupt(*args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(converter, "validate_video", interrupt)
    output = tmp_path / "changed_resume"
    with pytest.raises(RuntimeError):
        convert(subset, output, "train", contract, min_age=0, video_mode="copy")
    with pytest.raises(ValueError, match="configuration changed"):
        convert(
            subset, output, "train", dict(contract, joint_unit="different"), min_age=0, video_mode="copy", resume=True
        )
    (tmp_path / "unknown.incomplete").mkdir()
    with pytest.raises(ValueError, match="no matching resume checkpoint"):
        convert(subset, tmp_path / "unknown", "train", contract, min_age=0, video_mode="copy", resume=True)
    with conversion_lock(tmp_path / "locked.incomplete"), pytest.raises(RuntimeError, match="Another conversion"):
        convert(subset, tmp_path / "locked", "train", contract, min_age=0, video_mode="copy", resume=True)


def test_batch_runner_reuses_published_version(subset, contract, tmp_path):
    for path in subset.rglob("*"):
        if path.is_file():
            os.utime(path, (time.time() - 600, time.time() - 600))
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "version"
    command = [
        "bash",
        str(repo / "scripts/run_yam_conversion.sh"),
        sys.prefix,
        str(repo),
        str(subset),
        str(output),
        str(contract_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    before = {p: digest(p) for p in output.rglob("*") if p.is_file()}
    result = subprocess.run([*command, "--resume"], check=True, capture_output=True, text=True)
    assert "REUSED_COMPLETED_SPLIT=val" in result.stdout
    assert "REUSED_COMPLETED_SPLIT=train" in result.stdout
    assert all(digest(path) == value for path, value in before.items())
