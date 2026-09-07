import importlib.metadata
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "docker/thor/preserve_vendor_stack.py"


def test_pytorch_constraints_preserve_vendor_backends_not_optional_transformers(tmp_path, monkeypatch):
    packages = ["torch", "torchvision", "triton", "nvidia_modelopt", "tensorrt_cu13", "numpy", "transformers", "jax"]
    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": name}, version="1.0") for name in packages],
    )
    output = tmp_path / "constraints.txt"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(output), "--backend", "pytorch"])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert set(output.read_text().splitlines()) == {
        "torch==1.0",
        "torchvision==1.0",
        "triton==1.0",
        "nvidia-modelopt==1.0",
        "tensorrt-cu13==1.0",
        "numpy==1.0",
    }


def test_default_jax_constraints_remain_available(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0")
    output = tmp_path / "constraints.txt"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(output)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert "jax-cuda13-plugin==1.0" in output.read_text()
    assert "torch==" not in output.read_text()
