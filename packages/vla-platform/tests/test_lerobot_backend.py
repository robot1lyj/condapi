"""CPU wiring tests; these do not claim LeRobot or any model has run."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest
from test_platform import experiment
from test_platform import project  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("lerobot_launcher", ROOT / "adapters/lerobot/train.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize(
    ("model", "policy"),
    [("evo1", "evo1"), ("fastwam", "fastwam"), ("vla-jepa", "vla_jepa"), ("molmoact2", "molmoact2")],
)
def test_models_share_native_launcher(project, model, policy):  # noqa: F811
    # Explicit test-only promotion. Real model declarations remain planned.
    declaration = project.root / f"configs/models/{model}.toml"
    text = declaration.read_text().replace('status = "planned"', 'status = "implemented"')
    declaration.write_text(text.replace("contracts = []", 'contracts = ["yam-bimanual-v1"]\noperations = ["train"]'))
    profile = project.root / "configs/environments/pi-server.toml"
    profile.write_text(profile.read_text().replace('model = "pi"', f'model = "{model}"'))
    native = project.root / "native.json"
    native.write_text(json.dumps({"policy": {"type": policy}, "custom_upstream_setting": 42}))
    path = experiment(
        project,
        f"""schema_version = 1
model = "{model}"
environment = "configs/environments/pi-server.toml"
contract = "configs/robots/yam.toml"
[parameters]
native_config = "native.json"
""",
    )
    plan = project.plan(path, "train", "native-test")
    assert plan.implementation == "lerobot"
    assert str(project.root / "adapters/lerobot/train.py") in plan.command
    assert policy in plan.command
    assert str(native) in plan.source_hashes
    assert str(declaration) in plan.source_hashes
    assert str(project.root / "adapters/lerobot/backend.toml") in plan.source_hashes
    assert not Path(plan.output).exists()


def test_native_config_is_preserved_and_side_effects_overridden(tmp_path):
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "policy": {"type": "evo1", "use_amp": True, "push_to_hub": True},
                "wandb": {"enable": True},
                "job": {"target": "local"},
                "save_checkpoint_to_hub": True,
            }
        )
    )
    before = config.read_bytes()
    args = launcher.native_arguments(config, "evo1", tmp_path / "output")
    assert f"--config_path={config}" in args
    assert "--policy.push_to_hub=false" in args
    assert "--wandb.enable=false" in args
    assert "--save_checkpoint_to_hub=false" in args
    assert "--job.target=local" in args
    assert not any("amp" in arg or "dtype" in arg for arg in args)
    assert config.read_bytes() == before


@pytest.mark.parametrize(
    "extra",
    [
        {"resume": True},
        {"job": {"target": "a100-large"}},
        {"reward_model": {"type": "fixture"}},
        {"policy": {"type": "fastwam"}},
    ],
)
def test_launcher_rejects_wrong_workflow(tmp_path, extra):
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"policy": {"type": "evo1"}, **extra}))
    with pytest.raises(ValueError, match=r"match|new runs|local policy"):
        launcher.native_arguments(config, "evo1", tmp_path / "output")


def test_launcher_delegates_without_implementing_training(tmp_path, monkeypatch):
    config = tmp_path / "train.json"
    config.write_text('{"policy": {"type": "evo1"}}')
    calls = []
    original = sys.argv
    monkeypatch.setattr(launcher.runpy, "run_module", lambda module, **kw: calls.append((module, kw, sys.argv[:])))
    launcher.main(["--config", str(config), "--policy-type", "evo1", "--output", str(tmp_path / "output")])
    assert calls[0][0] == "lerobot.scripts.lerobot_train"
    assert calls[0][1] == {"run_name": "__main__"}
    assert calls[0][2][0] == "lerobot-train"
    assert sys.argv is original


def test_backend_and_model_capabilities_are_separate(project):  # noqa: F811
    assert "train" in project.backends()["lerobot"][1]["operations"]
    for model in ("evo1", "fastwam", "vla-jepa", "molmoact2"):
        assert project.models()[model][1]["status"] == "planned"
        assert not project.models()[model][1].get("operations")


def test_model_cannot_define_its_own_training_loop(project):  # noqa: F811
    declaration = project.root / "configs/models/pi.toml"
    declaration.write_text(
        declaration.read_text().replace(
            'operations = ["train", "infer", "benchmark", "export"]',
            '[operations.train]\ncommand = ["python", "custom.py"]',
        )
    )
    with pytest.raises(ValueError, match="capability list"):
        project.models()
