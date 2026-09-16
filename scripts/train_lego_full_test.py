import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import train_lego_full as entry


def test_full_config(monkeypatch):
    monkeypatch.setenv("LEGO_BATCH_SIZE", "64")
    monkeypatch.setenv("LEGO_SAVE_INTERVAL", "5000")
    monkeypatch.setenv("LEGO_KEEP_PERIOD", "5000")
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


def test_rtc_adaptation_config(tmp_path, monkeypatch):
    monkeypatch.delenv("LEGO_PEAK_LR", raising=False)
    monkeypatch.delenv("LEGO_DECAY_LR", raising=False)
    params = tmp_path / "100000" / "params"
    params.mkdir(parents=True)
    norm = params.parent / "assets" / "yam" / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    config = entry.make_config(init_params=str(params), rtc_training_max_delay=10, train_episodes=[2, 8], steps=5000)
    assert config.weight_loader.params_path == str(params)
    assert config.model.rtc_training_max_delay == 10
    assert config.data.base_config.train_episodes == [2, 8]
    assert config.lr_schedule.peak_lr == 3e-6
    assert config.lr_schedule.warmup_steps == 200
    assert config.lr_schedule.decay_steps == 5000
    assert not config.resume
    control = tmp_path / "control"
    control.mkdir()
    entry.check_training_contract(config, control)
    entry.check_training_contract(dataclasses.replace(config, resume=True), control)
    changed = dataclasses.replace(
        config, resume=True, model=dataclasses.replace(config.model, rtc_training_max_delay=0)
    )
    with pytest.raises(RuntimeError, match="contract changed"):
        entry.check_training_contract(changed, control)
    norm.write_text('{"modified": true}')
    with pytest.raises(RuntimeError, match="contract changed"):
        entry.check_training_contract(dataclasses.replace(config, resume=True), control)


@pytest.mark.parametrize("values", [[], [True], [-1], [1, 1], {}, [1.5]])
def test_invalid_episode_ids(tmp_path, values):
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="Episode"):
        entry.read_episode_ids(path)


def test_rtc_cannot_implicitly_start_from_base():
    with pytest.raises(ValueError, match="init-params"):
        entry.make_config(rtc_training_max_delay=10)


def test_base_rtc_with_new_subset_norm(tmp_path):
    params = tmp_path / "pi05_base" / "params"
    params.mkdir(parents=True)
    assets = tmp_path / "subset_norm"
    norm = assets / "yam" / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    provenance = norm.parent / "provenance.json"
    provenance.write_text(json.dumps({"selected_episode_ids": [2, 8]}))
    config = entry.make_config(
        init_params=str(params),
        assets_dir=str(assets),
        train_episodes=[2, 8],
        rtc_training_max_delay=10,
        steps=10000,
        warmup_steps=1000,
        decay_steps=30000,
    )
    assert config.data.assets.assets_dir == str(assets)
    assert config.lr_schedule.decay_steps == 30000
    assert config.weight_loader.params_path == str(params)
    assert not config.resume
    with pytest.raises(ValueError, match="Norm episode selection"):
        entry.make_config(init_params=str(params), assets_dir=str(assets), train_episodes=[2])
