import hashlib
import json
from types import SimpleNamespace

import pytest

from download_openwam_foundation import REPO_ID
from download_openwam_foundation import download


def test_download_pins_revision_resumes_and_writes_receipt(tmp_path, monkeypatch):
    payload = b"foundation"
    info = SimpleNamespace(
        sha="fixed-revision",
        siblings=[
            SimpleNamespace(
                rfilename="config.yaml",
                size=len(payload),
                lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()),
            )
        ],
    )
    calls = []

    def model_info(repo, **kwargs):
        assert repo == REPO_ID
        assert kwargs == {"revision": "main", "files_metadata": True}
        return info

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ConnectionError("interrupted")
        (tmp_path / "config.yaml").write_bytes(payload)

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=model_info), snapshot_download=snapshot_download),
    )
    receipt = download(tmp_path, retries=2)
    assert receipt["revision"] == "fixed-revision"
    assert calls[0]["revision"] == calls[1]["revision"] == "fixed-revision"
    assert json.loads((tmp_path / "foundation-download-receipt.json").read_text())["finetune_only"]


def test_download_rejects_revision_mix(tmp_path, monkeypatch):
    (tmp_path / "foundation-download-request.json").write_text(
        json.dumps({"repo": REPO_ID, "requested_revision": "old", "revision": "old-sha"})
    )
    info = SimpleNamespace(sha="new-sha", siblings=[])
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *a, **k: info), snapshot_download=lambda **k: None),
    )
    with pytest.raises(ValueError, match="different revision"):
        download(tmp_path, retries=1)
