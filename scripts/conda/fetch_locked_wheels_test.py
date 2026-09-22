import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("fetch_wheels", Path(__file__).with_name("fetch_locked_wheels.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def item(content=b"fixture"):
    return {"filename": "fixture.whl", "name": "lerobot", "sha256": hashlib.sha256(content).hexdigest()}


def test_source_wheel_copy_and_reuse(tmp_path):
    source = tmp_path / "source.whl"
    source.write_bytes(b"fixture")
    module.fetch(item(), tmp_path, source)
    source.unlink()
    module.fetch(item(), tmp_path, source)
    assert (tmp_path / "fixture.whl").read_bytes() == b"fixture"


def test_wrong_hash_cannot_publish(tmp_path):
    source = tmp_path / "source.whl"
    source.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.fetch(item(), tmp_path, source)
    assert not (tmp_path / "fixture.whl").exists()


def test_basename_cannot_escape(tmp_path):
    value = {**item(), "filename": "../escape.whl"}
    with pytest.raises(ValueError, match="basename"):
        module.fetch(value, tmp_path, tmp_path / "source")


def test_partial_cannot_overwrite_symlink_target(tmp_path):
    source = tmp_path / "source.whl"
    source.write_bytes(b"fixture")
    (tmp_path / "fixture.whl.fetch-part").symlink_to(source)
    with pytest.raises(ValueError, match="symlinks"):
        module.fetch(item(), tmp_path, source)
    assert source.read_bytes() == b"fixture"


def test_official_cuda_wheel_via_optional_proxy(tmp_path, monkeypatch):
    calls = []

    def download(args, *, check):
        calls.append(args)
        Path(args[args.index("--output") + 1]).write_bytes(b"fixture")
        assert check

    monkeypatch.setattr(module.subprocess, "run", download)
    value = {**item(), "name": "torch", "url": "https://download.pytorch.org/whl/cu128/fixture.whl"}
    module.fetch(value, tmp_path, None, "socks5h://127.0.0.1:18891")
    assert calls[0][-1] == value["url"]
    assert calls[0][1:3] == ["--proxy", "socks5h://127.0.0.1:18891"]
    assert (tmp_path / "fixture.whl").read_bytes() == b"fixture"


def test_pypi_mirror_default_unchanged(tmp_path, monkeypatch):
    calls = []

    def download(args, *, check):
        calls.append(args)
        Path(args[args.index("--output") + 1]).write_bytes(b"fixture")

    monkeypatch.setattr(module.subprocess, "run", download)
    value = {**item(), "name": "numpy", "url": "https://files.pythonhosted.org/packages/a/fixture.whl"}
    module.fetch(value, tmp_path, None)
    assert calls[0][-1] == "https://mirrors.aliyun.com/pypi/packages/a/fixture.whl"
    assert "--proxy" not in calls[0]
