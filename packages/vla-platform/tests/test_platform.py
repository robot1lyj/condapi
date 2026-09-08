import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from vla_platform.artifacts import REQUIRED_ROLES
from vla_platform.artifacts import seal_bundle
from vla_platform.artifacts import validate_bundle
from vla_platform.cli import main
from vla_platform.contracts import validate_request
from vla_platform.contracts import validate_response
from vla_platform.project import Project
from vla_platform.project import inside
from vla_platform.project import read_toml
from vla_platform.project import render
import vla_platform.runtime as runtime
from vla_platform.runtime import execute

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def project(tmp_path):
    for path in ("vla.toml", "configs", "adapters", "environments"):
        source = ROOT / path
        if source.is_dir():
            shutil.copytree(source, tmp_path / path)
        else:
            shutil.copy2(source, tmp_path / path)
    # Only source provenance needs these files; no model code is executed.
    for path in (
        "scripts/thor/benchmark_suite.py",
        "scripts/thor/export_pi05_onnx.py",
        "scripts/thor/maxn_session.py",
        "scripts/train.py",
    ):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# test source\n")
    return Project(tmp_path)


def experiment(project, text=None):
    path = project.root / "configs/experiments/pi-reference.toml"
    if text is not None:
        path.write_text(text)
    return path


def test_registry_does_not_import_model_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vla_platform.cli,sys; assert not {'torch','jax','numpy','transformers'} & sys.modules.keys()",
        ],
        env={"PYTHONPATH": str(ROOT / "packages/vla-platform/src")},
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_plan_is_read_only_and_uses_conda(project):
    plan = project.plan(experiment(project), "infer", "test-run")
    assert plan.command[:4] == ["conda", "run", "--no-capture-output", "--prefix"]
    assert not Path(plan.output).exists()
    assert "--compute" in plan.command
    assert plan.contract_id == "yam-bimanual-v1"


@pytest.mark.parametrize("name", ["evo1", "vla-jepa", "fastwam"])
def test_unimplemented_models_fail_before_execution(project, name):
    path = experiment(project)
    path.write_text(path.read_text().replace('model = "pi"', f'model = "{name}"'))
    with pytest.raises(ValueError, match="planned"):
        project.plan(path, "infer", "r1")


@pytest.mark.parametrize("name", ["../escape", "/tmp/root", "a/b", "", "a;rm"])
def test_run_identifiers_cannot_escape(project, name):
    with pytest.raises(ValueError, match="identifier"):
        project.plan(experiment(project), "infer", name)


def test_wrong_precision_rejected(project):
    path = experiment(project)
    path.write_text(path.read_text().replace('compute = "float32"', 'compute = "fp8"'))
    with pytest.raises(ValueError, match="Unsupported compute"):
        project.plan(path, "infer", "r1")


def test_shell_metacharacters_remain_single_argument():
    assert render("{path}", {"path": "/tmp/model; touch /tmp/unwanted"}) == "/tmp/model; touch /tmp/unwanted"


@pytest.mark.parametrize("template", ["{missing}", "{x.__class__}", "{x!r}", "{x:>20}"])
def test_template_rejects_attribute_access_and_unknown_fields(template):
    with pytest.raises(ValueError, match="placeholder"):
        render(template, {"x": "hello"})


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        inside(tmp_path, "escape/file")


def test_environment_family_mismatch(project):
    path = project.root / "configs/environments/pi-workstation.toml"
    path.write_text(path.read_text().replace('model = "pi"', 'model = "evo1"'))
    with pytest.raises(ValueError, match="another model family"):
        project.plan(experiment(project), "infer", "r1")


def test_control_side_target_rejected(project):
    path = project.root / "configs/environments/pi-workstation.toml"
    path.write_text(path.read_text().replace('target = "workstation"', 'target = "3588"'))
    with pytest.raises(ValueError, match="target"):
        project.plan(experiment(project), "infer", "r1")


def test_source_change_cannot_execute_stale_plan(project):
    path = project.root / "configs/environments/pi-workstation.toml"
    prefix = project.root / "envs/pi"
    (prefix / "conda-meta").mkdir(parents=True)
    path.write_text(path.read_text().replace("/home/wuyan-lyj/.conda/envs/vla-pi-dev", str(prefix)))
    plan = project.plan(experiment(project), "infer", "r1")
    experiment(project).write_text("# changed")
    with pytest.raises(ValueError, match="changed since planning"):
        execute(plan)
    assert not Path(plan.output).exists()


def test_second_backend_uses_same_core(project):
    directory = project.root / "adapters/test-double"
    directory.mkdir()
    (directory / "backend.toml").write_text("""schema_version = 1
id = "test-double"
status = "implemented"
contracts = ["yam-bimanual-v1"]
[operations.infer]
targets = ["workstation"]
command = ["python", "-c", "print(42)"]
""")
    (project.root / "configs/models/test-double.toml").write_text("""schema_version = 1
id = "test-double"
backend = "test-double"
status = "implemented"
contracts = ["yam-bimanual-v1"]
operations = ["infer"]
""")
    profile = project.root / "configs/environments/pi-workstation.toml"
    profile.write_text(profile.read_text().replace('model = "pi"', 'model = "test-double"'))
    path = experiment(project)
    path.write_text(path.read_text().replace('model = "pi"', 'model = "test-double"'))
    assert project.plan(path, "infer", "r1").command[-1] == "print(42)"


@pytest.fixture
def contract():
    return read_toml(ROOT / "configs/robots/yam.toml")


@pytest.fixture
def observation_request(contract):
    return {
        "schema_version": 1,
        "contract_id": contract["id"],
        "request_id": "r1",
        "session_id": "episode-1",
        "prompt": "pick up",
        "timestamp_s": 1.0,
        "state": [0.0] * 14,
        "images": dict.fromkeys(contract["cameras"], "rgb.png"),
    }


def test_valid_request(observation_request, contract):
    assert validate_request(observation_request, contract) is observation_request


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1"])
def test_non_numeric_or_nonfinite_state_rejected(observation_request, contract, value):
    observation_request["state"][0] = value
    with pytest.raises(ValueError, match="finite"):
        validate_request(observation_request, contract)


def test_missing_camera_rejected(observation_request, contract):
    observation_request["images"].pop(contract["cameras"][0])
    with pytest.raises(ValueError, match="Camera"):
        validate_request(observation_request, contract)


def test_response_semantics_and_dimension(contract):
    response = {
        "schema_version": 1,
        "contract_id": contract["id"],
        "request_id": "r1",
        "session_id": "s1",
        "model_version": "test",
        "actions": [[0.0] * 14] * 50,
        "action_dt_s": 0.1,
        "latency_ms": 10,
        "output_semantics": "absolute",
    }
    validate_response(response, contract)
    response["actions"] = [[0.0] * 7]
    with pytest.raises(ValueError, match="dimension"):
        validate_response(response, contract)


@pytest.fixture
def recipe(tmp_path):
    files = []
    for role in sorted(REQUIRED_ROLES):
        file = tmp_path / f"{role}.bin"
        if role == "contract":
            file.write_bytes((ROOT / "configs/robots/yam.toml").read_bytes())
        else:
            file.write_bytes(b"test-fixture-not-a-model")
        files.append({"role": role, "path": file.name})
    manifest = {
        "schema_version": 1,
        "model": "pi",
        "model_version": "fixture",
        "code_revision": "fixture",
        "contract_id": "yam-bimanual-v1",
        "format": "jax",
        "precision": {"storage": "float32", "compute": "bfloat16"},
        "files": files,
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(manifest))
    return path


def test_bundle_roundtrip_and_tamper(recipe):
    output = recipe.parent / "manifest.json"
    assert "not_accuracy" in seal_bundle(recipe, output)["status"]
    (recipe.parent / "weights.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle(output)


def test_bundle_no_overwrite(recipe):
    output = recipe.parent / "manifest.json"
    seal_bundle(recipe, output)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        seal_bundle(recipe, output)
    assert output.read_bytes() == before


def test_lora_requires_base_identity(recipe):
    value = json.loads(recipe.read_text())
    value["format"] = "adapter"
    recipe.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="base model"):
        seal_bundle(recipe, recipe.parent / "manifest.json")
    assert not (recipe.parent / "manifest.json").exists()


def test_incomplete_bundle_not_published(recipe):
    value = json.loads(recipe.read_text())
    value["files"].pop()
    recipe.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="handoff roles"):
        seal_bundle(recipe, recipe.parent / "manifest.json")
    assert not (recipe.parent / "manifest.json").exists()


def test_cli_models(capsys):
    assert main(["--root", str(ROOT), "models"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert {x["id"] for x in result} == {"pi", "evo1", "vla-jepa", "fastwam"}


def test_env_plan_no_install(project):
    result = project.environment_plan("configs/environments/pi-workstation.toml")
    assert result["command"][:3] == ["conda", "env", "create"]
    assert result["status"] == "bootstrap_not_model_ready"


def test_training_not_allowed_on_workstation(project):
    path = experiment(project)
    with pytest.raises(ValueError, match="target"):
        project.plan(path, "train", "r1")


@pytest.mark.parametrize("code", [0, 7])
def test_real_subprocess_records_exit_without_accepting_model(project, monkeypatch, code):
    # Exercise process/run-record implementation, not Conda or a real model.
    prefix = project.root / "envs/pi"
    (prefix / "conda-meta").mkdir(parents=True)
    profile = project.root / "configs/environments/pi-workstation.toml"
    profile.write_text(profile.read_text().replace("/home/wuyan-lyj/.conda/envs/vla-pi-dev", str(prefix)))
    plan = project.plan(experiment(project), "infer", "run-process")
    plan = dataclasses.replace(plan, command=[sys.executable, "-c", f"print('worker'); raise SystemExit({code})"])
    monkeypatch.setattr(runtime, "git_state", lambda root: {"commit": "test-fixture"})
    assert execute(plan) == code
    output = Path(plan.output)
    assert "worker" in (output / "console.log").read_text()
    finished = json.loads((output / "finished.json").read_text())
    assert finished["exit_code"] == code
    assert finished["status"] == ("command_succeeded_not_model_accepted" if code == 0 else "failed")
    with pytest.raises(FileExistsError):
        execute(plan)


def test_thor_profile_cannot_execute_on_workstation(project):
    path = experiment(project)
    path.write_text(path.read_text().replace("pi-workstation.toml", "pi-thor.toml"))
    prefix = project.root / "envs/pi"
    (prefix / "conda-meta").mkdir(parents=True)
    profile = project.root / "configs/environments/pi-thor.toml"
    profile.write_text(profile.read_text().replace("/home/wuyan-lyj/.conda/envs/vla-pi-infer", str(prefix)))
    plan = project.plan(path, "infer", "run-thor")
    with pytest.raises(ValueError, match="Thor profile"):
        execute(plan)
