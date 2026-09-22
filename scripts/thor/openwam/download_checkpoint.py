"""Download an immutable official deploy checkpoint, without touching CUDA."""

import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

repo = "OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full"
root = Path(os.environ.get("OPENWAM_CHECKPOINT_ROOT", "/assets/OpenWAM-Alpha-Sim-RoboTwin-Full"))
root.mkdir(parents=True, exist_ok=True)
request_path = root / "thor-download-request.json"
requested_revision = json.loads(request_path.read_text())["revision"] if request_path.exists() else "main"
info = HfApi().model_info(repo, revision=requested_revision, files_metadata=True)
revision = info.sha
request_path.write_text(json.dumps({"repo": repo, "revision": revision}) + "\n")
print(json.dumps({"repo": repo, "revision": revision}), flush=True)
for attempt in range(20):
    try:
        snapshot_download(repo, revision=revision, local_dir=root, max_workers=2)
        break
    except Exception as exc:
        print(f"Download attempt {attempt + 1} failed: {type(exc).__name__}: {exc}", flush=True)
        if attempt == 19:
            raise
        time.sleep(10)
files = []
for entry in info.siblings:
    path = root / entry.rfilename
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = getattr(entry.lfs, "sha256", None) if entry.lfs else None
    if expected and digest != expected:
        raise RuntimeError(f"SHA256 mismatch: {entry.rfilename}")
    files.append({"file": entry.rfilename, "size": path.stat().st_size, "sha256": digest})
(root / "thor-download-receipt.json").write_text(
    json.dumps({"repo": repo, "revision": revision, "files": files}, indent=2) + "\n"
)
print("DOWNLOAD_VERIFIED", flush=True)
