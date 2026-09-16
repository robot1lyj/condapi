"""Content identity for original RTC Orbax parameters (not metadata-only)."""

import hashlib


def source_params_sha256(checkpoint):
    root = checkpoint / "params"
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Original RTC parameter files are missing")
    result = {}
    for path in files:
        with path.open("rb") as stream:
            result[str(path.relative_to(checkpoint))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result
