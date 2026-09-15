"""Host command tests with mocked GPU execution; no model or training loop."""

import json
from pathlib import Path
import sys

import pytest

from scripts.thor import run_export_host
from scripts.thor import run_suite_host


@pytest.mark.parametrize("module", [run_suite_host, run_export_host])
def test_checkpoint_paths_and_offline_tokenizer(monkeypatch, tmp_path, module):
    root = tmp_path / "pi"
    root.mkdir()
    args = [
        "runner",
        "--root",
        str(root),
        "--image",
        "pi:test",
        "--code-commit",
        "test",
        "--checkpoint",
        "lego/100000-pytorch-fp32",
        "--suite",
        "new/suite.json",
    ]
    if module is run_suite_host:
        args += ["--batch-id", "test", "--modes", "D"]
    else:
        args += ["--run-id", "test", "--prepare-only"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "mock\n")
    monkeypatch.setattr(module.subprocess, "call", lambda *a, **k: 0)
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **k: "NVIDIA Thor" if str(p) == "/proc/device-tree/model" else original(p, *a, **k),
    )
    if module is run_export_host:
        with pytest.raises(SystemExit) as result:
            module.main()
        assert result.value.code == 0
    else:
        module.main()
    manifest = json.loads(next((root / "logs").glob("*.manifest.json")).read_text())
    command = manifest["command"]
    assert command[command.index("--checkpoint") + 1] == "/checkpoints/lego/100000-pytorch-fp32"
    assert command[command.index("--suite") + 1] == "/test-data/new/suite.json"
    assert "OPENPI_DATA_HOME=/cache" in command
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0" in command
    assert command[command.index("--network") + 1] == "none"


@pytest.mark.parametrize("module", [run_suite_host, run_export_host])
@pytest.mark.parametrize("path", ["../old", "/absolute"])
def test_reject_path_escape(monkeypatch, module, path):
    args = ["runner", "--image", "pi:test", "--code-commit", "test", "--checkpoint", path]
    args += ["--batch-id", "test"] if module is run_suite_host else ["--run-id", "test"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 2
