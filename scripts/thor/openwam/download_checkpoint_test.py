import hashlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).with_name("download_checkpoint.py")


@pytest.mark.parametrize("corrupt", [False, True])
def test_download_pins_revision_retries_and_checks_hash(tmp_path, monkeypatch, corrupt):
    calls = []
    payload = b"test-checkpoint"
    expected = "0" * 64 if corrupt else hashlib.sha256(payload).hexdigest()
    info = SimpleNamespace(
        sha="fixed-revision",
        siblings=[SimpleNamespace(rfilename="weights", lfs=SimpleNamespace(sha256=expected))],
    )

    def model_info(repo, **kwargs):
        assert repo == "OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full"
        assert kwargs["files_metadata"]
        return info

    def snapshot(repo, **kwargs):
        calls.append(kwargs["revision"])
        if len(calls) == 1:
            raise ConnectionError("simulated interrupted transfer")
        (tmp_path / "weights").write_bytes(payload)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=model_info), snapshot_download=snapshot),
    )
    monkeypatch.setenv("OPENWAM_CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setattr("time.sleep", lambda _: None)
    if corrupt:
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            runpy.run_path(str(SCRIPT))
        assert not (tmp_path / "thor-download-receipt.json").exists()
    else:
        runpy.run_path(str(SCRIPT))
        assert (tmp_path / "thor-download-receipt.json").exists()
    assert calls == ["fixed-revision", "fixed-revision"]
