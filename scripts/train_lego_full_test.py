from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import train_lego_full as entry


def test_full_config():
    config = entry.make_config(run_name="lego_full_b64_r2_20260909")
    assert config.save_interval == config.keep_period == 5000
    assert config.num_train_steps == 40000
    assert config.batch_size == 64
    assert config.fsdp_devices == 4
    assert config.ema_decay is None
    assert not config.resume
    assert not config.overwrite
    assert config.exp_name == "lego_full_b64_r2_20260909"


@pytest.mark.parametrize("committed", [False, True])
def test_resume_requires_committed_checkpoint(tmp_path, monkeypatch, committed):
    manager = Mock()
    initialize = Mock(return_value=(manager, committed))
    monkeypatch.setattr(entry.checkpoints, "initialize_checkpoint_dir", initialize)
    config = SimpleNamespace(checkpoint_dir=tmp_path, keep_period=5000)
    if committed:
        entry.require_resume_checkpoint(config)
    else:
        with pytest.raises(RuntimeError, match="refusing to restart from base"):
            entry.require_resume_checkpoint(config)
    manager.close.assert_called_once()
    initialize.assert_called_once_with(tmp_path, keep_period=5000, overwrite=False, resume=True)


def test_missing_resume_does_not_create_directory(tmp_path):
    path = tmp_path / "absent"
    with pytest.raises(RuntimeError, match="missing"):
        entry.require_resume_checkpoint(SimpleNamespace(checkpoint_dir=path))
    assert not path.exists()
