import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.thor import download_ngc_image as download


def test_reuses_verified_cache_without_fetching_blobs(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    payload = b"synthetic image blob"
    name = hashlib.sha256(payload).hexdigest()
    (cache / name).write_bytes(payload)
    item = {"digest": f"sha256:{name}", "size": len(payload)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"config": item, "layers": [item]}))
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download",
            "--manifest",
            str(manifest),
            "--manifest-digest",
            f"sha256:{download.digest(manifest)}",
            "--cache",
            str(cache),
            "--output",
            str(output),
        ],
    )
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        assert url == "https://nvcr.io/proxy_auth"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"token": "synthetic"})

    monkeypatch.setattr(download.requests, "get", get)
    download.main()
    assert len(calls) == 0
    assert (output / name).read_bytes() == payload
    assert (cache / name).read_bytes() == payload
    assert (output / "manifest.json").read_bytes() == manifest.read_bytes()


@pytest.mark.parametrize("value", ["sha256:../escape", "sha256:abcd", "md5:" + "0" * 64])
def test_invalid_blob_name_rejected(value):
    with pytest.raises(ValueError, match="Only SHA-256"):
        download.checked_name(value)
