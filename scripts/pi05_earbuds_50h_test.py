import json
import sys

import pi05_earbuds_50h as entry
import pytest


def test_frozen_selection_uses_only_train_and_separate_id_namespaces():
    recipe, report, selection = entry.checked_recipe(entry.DEFAULT_RECIPE)
    assert len(selection["selected_episodes"]) == 1384
    assert selection["frames"] == 5402194
    assert all(row["source_repo"] == "lerobot/abc_130k_v3_train" for row in selection["selected_episodes"])
    assert entry.read_json(report / "episodes.json") == list(range(1384))
    assert entry.read_json(report / "source-episodes.json") != list(range(1384))
    assert recipe["automatic_training_start"] is False


def test_training_command_keeps_task_assets_and_lr_across_stops(monkeypatch):
    monkeypatch.setenv("LEGO_PEAK_LR", "999")
    recipe, _, _ = entry.checked_recipe(entry.DEFAULT_RECIPE)
    assert recipe["training"]["first_stop_step"] == 200000
    first, environment = entry.train_command(recipe, recipe["training"]["first_stop_step"])
    assert first[first.index("--steps") + 1] == "200000"
    resumed, resumed_environment = entry.train_command(recipe, 338000, resume=True)
    assert first[first.index("--dataset") + 1] == recipe["dataset_root"] + "/train"
    assert first[first.index("--assets-dir") + 1] == recipe["assets_root"] + "/norm"
    assert first[first.index("--init-params") + 1] == recipe["init_params"]
    assert resumed[resumed.index("--decay-steps") + 1] == "338000"
    assert "--resume" not in first
    assert "--resume" in resumed
    assert environment["LEGO_PEAK_LR"] == "1e-05"
    assert resumed_environment == environment


@pytest.mark.parametrize("step", [0, -1, 338001])
def test_invalid_cumulative_stop_is_rejected(step):
    recipe, _, _ = entry.checked_recipe(entry.DEFAULT_RECIPE)
    with pytest.raises(ValueError, match="Cumulative stop"):
        entry.train_command(recipe, step)


def test_recipe_check_never_prepares_or_trains(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Recipe-only validation must not execute a subprocess")

    monkeypatch.setattr(entry.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["pi05_earbuds_50h.py", "check", "--recipe-only"])
    entry.main()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "RECIPE_READY_DATA_UNVERIFIED"
    assert output["automatic_training_start"] is False


def test_data_preparation_requires_compute_allocation(monkeypatch):
    recipe, report, selection = entry.checked_recipe(entry.DEFAULT_RECIPE)
    monkeypatch.setattr(entry, "require_server", lambda recipe: None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute-node Slurm"):
        entry.prepare_data(recipe, report, selection)


def test_training_requires_explicit_slurm_job(monkeypatch):
    recipe, report, selection = entry.checked_recipe(entry.DEFAULT_RECIPE)
    monkeypatch.setattr(entry, "require_server", lambda recipe: None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="explicitly submitted"):
        entry.launch(recipe, report, selection, 170000)


def test_pending_data_blocks_training_before_launch(monkeypatch):
    recipe, report, selection = entry.checked_recipe(entry.DEFAULT_RECIPE)
    monkeypatch.setattr(entry, "require_server", lambda recipe: None)
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")

    def pending(*args):
        raise FileNotFoundError("Earbuds norm not prepared")

    def forbidden(*args, **kwargs):
        pytest.fail("Unprepared data must never reach the native trainer")

    monkeypatch.setattr(entry, "check_data", pending)
    monkeypatch.setattr(entry.subprocess, "run", forbidden)
    with pytest.raises(FileNotFoundError, match="norm not prepared"):
        entry.launch(recipe, report, selection, 170000)
