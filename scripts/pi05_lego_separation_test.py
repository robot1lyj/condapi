"""Approved recipe and native command checks; no model/optimizer construction."""

import copy

import pytest

from scripts import pi05_lego_separation as entry


def test_frozen_approved_data_and_mixture():
    recipe, _, data = entry.checked_recipe(entry.DEFAULT_RECIPE)
    assert recipe["approval"]["training_authorized"] is True
    assert len(data["episodes"]) == 200
    assert data["pool_stats"]["specialist"]["frames"] == data["pool_stats"]["replay"]["frames"] == 278516


def test_command_uses_parent_norm_and_fixed_new_run_budget(monkeypatch):
    monkeypatch.setenv("LEGO_PEAK_LR", "999")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    recipe, _, _ = entry.checked_recipe(entry.DEFAULT_RECIPE)
    command, env = entry.train_command(recipe)
    resumed, resumed_env = entry.train_command(recipe, resume=True)
    assert command[command.index("--steps") + 1] == "5000"
    assert command[command.index("--init-params") + 1].endswith("/240000/params")
    assert command[command.index("--train-episodes-file") + 1] == recipe["selection_root"] + "/episodes.json"
    assert "--assets-dir" not in command  # Inherit the parent's norm; do not recompute subset norm.
    assert "--resume" not in command
    assert "--resume" in resumed
    assert resumed_env == env
    assert env["LEGO_PEAK_LR"] == "3e-06"
    assert "JAX_PLATFORMS" not in env


def test_changed_recipe_is_rejected(monkeypatch):
    original = entry.read_json
    altered = copy.deepcopy(original(entry.DEFAULT_RECIPE))
    altered["training"]["peak_lr"] = 1e-5
    monkeypatch.setattr(entry, "read_json", lambda path: altered if path == entry.DEFAULT_RECIPE else original(path))
    with pytest.raises(ValueError, match="explicitly confirmed"):
        entry.checked_recipe(entry.DEFAULT_RECIPE)


def test_launch_needs_server_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    recipe, report, data = entry.checked_recipe(entry.DEFAULT_RECIPE)
    with pytest.raises(RuntimeError, match="Slurm"):
        entry.launch(recipe, report, data)


def test_10k_replacement_has_independent_run_and_matching_decay():
    old, _, _ = entry.checked_recipe(entry.DEFAULT_RECIPE)
    new, _, _ = entry.checked_recipe(entry.RECIPE_10K)
    command, _ = entry.train_command(new)
    assert new["run_name"] != old["run_name"]
    assert new["control_root"] != old["control_root"]
    assert new["selection_sha256"] == old["selection_sha256"]
    assert new["parent_checkpoint"] == old["parent_checkpoint"]
    assert command[command.index("--steps") + 1] == "10000"
    assert command[command.index("--decay-steps") + 1] == "10000"
    assert command[command.index("--warmup-steps") + 1] == "200"
    assert "--resume" not in command


def test_10k_cannot_keep_5k_decay(monkeypatch):
    original = entry.read_json
    altered = copy.deepcopy(original(entry.RECIPE_10K))
    altered["training"]["decay_steps"] = 5000
    monkeypatch.setattr(entry, "read_json", lambda path: altered if path == entry.RECIPE_10K else original(path))
    with pytest.raises(ValueError, match="explicitly confirmed"):
        entry.checked_recipe(entry.RECIPE_10K)
