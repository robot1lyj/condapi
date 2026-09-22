"""Resume and verify the official OpenWAM-Alpha fine-tuning checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

REPO_ID = "OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model"
VIDEO_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest(api, repo_id: str, revision: str):
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    return info, info.sha


def download(
    root: Path,
    requested_revision: str = "main",
    retries: int = 20,
    *,
    repo_id: str = REPO_ID,
    request_name: str = "foundation-download-request.json",
    receipt_name: str = "foundation-download-receipt.json",
    finetune_only: bool = True,
) -> dict:
    """Download one immutable revision into ``root`` and write a receipt.

    ``snapshot_download`` is resumable.  The request file prevents a later
    invocation from silently mixing a different revision into an existing
    checkpoint directory.
    """

    from huggingface_hub import HfApi  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / request_name
    prior = json.loads(request_path.read_text()) if request_path.exists() else None
    if prior and prior.get("repo") != repo_id:
        raise ValueError(f"Existing request belongs to {prior.get('repo')!r}")
    requested = prior.get("requested_revision", requested_revision) if prior else requested_revision
    info, revision = _manifest(HfApi(), repo_id, requested)
    if prior and prior.get("revision") not in (None, revision):
        raise ValueError("Existing checkpoint request resolves to a different revision")
    request_path.write_text(
        json.dumps(
            {
                "repo": repo_id,
                "requested_revision": requested,
                "revision": revision,
                "started_at": prior.get("started_at") if prior else time.time(),
            },
            indent=2,
        )
        + "\n"
    )

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(root),
                max_workers=2,
            )
            break
        except Exception as exc:  # keep partial files for the next attempt
            last_error = exc
            print(f"download attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(min(60, 5 * attempt))
    if last_error is not None:
        print(f"download resumed after transient error: {type(last_error).__name__}", flush=True)

    files = []
    for entry in info.siblings:
        path = root / entry.rfilename
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = getattr(entry, "size", None)
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"Size mismatch for {entry.rfilename}")
        digest = _sha256(path)
        expected_hash = getattr(entry.lfs, "sha256", None) if entry.lfs else None
        if expected_hash and digest != expected_hash:
            raise ValueError(f"SHA256 mismatch for {entry.rfilename}")
        files.append({"file": entry.rfilename, "size": path.stat().st_size, "sha256": digest})

    receipt = {
        "repo": repo_id,
        "requested_revision": requested,
        "revision": revision,
        "files": files,
        "finetune_only": finetune_only,
        "dtype_audit": "not_run",
        "environment": {key: os.environ[key] for key in ("HF_HUB_DISABLE_XET",) if key in os.environ},
    }
    receipt_path = root / receipt_name
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--request-name", default="foundation-download-request.json")
    parser.add_argument("--receipt-name", default="foundation-download-receipt.json")
    parser.add_argument("--finetune-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    receipt = download(
        args.root,
        args.revision,
        repo_id=args.repo,
        request_name=args.request_name,
        receipt_name=args.receipt_name,
        finetune_only=args.finetune_only,
    )
    print(json.dumps({"repo": receipt["repo"], "revision": receipt["revision"], "files": len(receipt["files"])}, indent=2))


if __name__ == "__main__":
    main()
