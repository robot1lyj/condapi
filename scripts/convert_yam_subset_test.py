import json

import av
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


def test_roundtrip_preserves_values_and_split(subset, contract, tmp_path):
    before = {p: digest(p) for p in subset.rglob("*") if p.is_file()}
    for split in ("train", "val"):
        output = convert(subset, tmp_path / f"clean_{split}", split, contract, min_age=0)
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


def test_changed_source_never_published(subset, contract, tmp_path, monkeypatch):
    original_save = LeRobotDataset.save_episode

    def changed_save(self, *args, **kwargs):
        result = original_save(self, *args, **kwargs)
        path = subset / "manifests/train.jsonl"
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(LeRobotDataset, "save_episode", changed_save)
    with pytest.raises(ValueError, match="Source changed"):
        convert(subset, tmp_path / "changed", "train", contract, min_age=0)
    assert not (tmp_path / "changed").exists()
    assert (tmp_path / "changed.incomplete").exists()
