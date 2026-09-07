import dataclasses
import json

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training import config
from openpi.training import data_loader
from scripts.audit_yam_subset import audit
from scripts.audit_yam_subset import validate_rows
from scripts.compute_norm_stats import _default_output_dir


def test_upload_missing_files_is_pending(tmp_path):
    (tmp_path / "manifests").mkdir()
    episode = {
        "source_episode_index": 95,
        "source_repo": "train",
        "source_revision": "abc",
        "length": 3,
        "fps": 30,
        "task": "sort legos",
        "data": {"file_index": 0},
    }
    (tmp_path / "manifests/train.jsonl").write_text(json.dumps(episode) + "\n")
    result = audit(tmp_path, inventory_only=True)
    assert result["counts"] == {"pending_upload": 1}
    assert len(result["episodes"][0]["missing"]) == 4
    assert result["trainable"] is False


def test_rows_reject_nonfinite_and_time_mismatch():
    rows = {
        "observation.state": np.zeros((3, 14)).tolist(),
        "action": np.ones((3, 14)).tolist(),
        "frame_index": [0, 1, 2],
        "timestamp": [0, 1 / 30, 2 / 30],
    }
    episode = {"length": 3, "fps": 30, "task": "sort legos"}
    validate_rows(pa.table(rows), episode)
    rows["action"][1][3] = float("nan")
    with pytest.raises(ValueError, match="invalid_action"):
        validate_rows(pa.table(rows), episode)
    rows["action"][1][3] = 1
    rows["timestamp"][2] = 1
    with pytest.raises(ValueError, match="timestamp_mismatch"):
        validate_rows(pa.table(rows), episode)


def test_yam_norm_default_matches_training_assets(tmp_path):
    cfg = config.get_config("pi05_yam_lora")
    data = config.DataConfig(repo_id=str(tmp_path / "dataset"), asset_id="yam")
    assert _default_output_dir(data, cfg) == cfg.assets_dirs / "yam"
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, assets=config.AssetsConfig(assets_dir=str(tmp_path / "assets"), asset_id="yam")
        ),
    )
    assert _default_output_dir(data, cfg) == tmp_path / "assets/yam"


def test_full_audit_decodes_videos_and_rejects_truncation(tmp_path):
    test_upload_missing_files_is_pending(tmp_path)
    data_dir = tmp_path / "train/data"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [95] * 3,
                "frame_index": [0, 1, 2],
                "timestamp": [0, 1 / 30, 2 / 30],
                "observation.state": np.zeros((3, 14)).tolist(),
                "action": np.ones((3, 14)).tolist(),
            }
        ),
        data_dir / "source-file-000.parquet",
    )
    for camera in ("top", "left_wrist", "right_wrist"):
        path = tmp_path / "train/videos" / camera / "episode-000095.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for _ in range(3):
                frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    assert audit(tmp_path, min_age=0)["counts"] == {"validated_structure": 1}
    path.write_bytes(b"broken video")
    assert audit(tmp_path, min_age=0)["counts"] == {"rejected": 1}


def test_raw_local_dataset_rejected_before_hub_access(tmp_path):
    cfg = config.get_config("pi05_yam_lora")
    with pytest.raises(ValueError, match="not a published LeRobot"):
        data_loader.create_torch_dataset(config.DataConfig(repo_id=str(tmp_path)), 50, cfg.model)
