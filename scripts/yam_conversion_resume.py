"""Durable, scoped checkpoints for copy-mode dataset conversion (Linux/NFS)."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def sync_file(path):
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".tmp-", delete=False) as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)
    sync_directory(path.parent)


def quarantine(path):
    """Move only an explicitly identified derived artifact; preserve for inspection."""
    path = Path(path)
    target = path.with_name(path.name + f".interrupted-{time.time_ns()}")
    path.rename(target)
    sync_directory(path.parent)
    return target


@contextmanager
def conversion_lock(work):
    work.parent.mkdir(parents=True, exist_ok=True)
    with work.with_name(work.name + ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another conversion owns this output") from None
        yield


def initialize_checkpoint(work, identity, resume):
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    marker = work / "resume_identity.json"
    if work.exists():
        if not resume or not marker.is_file():
            raise ValueError("Incomplete output has no matching resume checkpoint; do not adopt automatically")
        if json.loads(marker.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Resume source/configuration changed; use a new output version")
    else:
        work.mkdir(parents=True)
        atomic_json(marker, {"fingerprint": fingerprint, "identity": identity})
    return work / "resume_records"
