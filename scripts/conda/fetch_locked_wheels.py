"""Fetch exact wheel artifacts with curl, then verify SHA256 before offline pip install."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(item, directory, source_wheel):
    name = item["filename"]
    if Path(name).name != name or not name.endswith(".whl"):
        raise ValueError("Expected a wheel basename")
    target = directory / name
    partial = directory / f"{name}.fetch-part"
    if target.is_symlink() or partial.is_symlink():
        raise ValueError("Wheel cache paths must not be symlinks")
    if target.exists() and digest(target) == item["sha256"]:
        print(f"verified existing: {name}", flush=True)
        return
    if item["name"] == "lerobot":
        shutil.copyfile(source_wheel, partial)
    else:
        source = urlsplit(item["url"])
        if source.scheme != "https" or not source.path.startswith("/packages/"):
            raise ValueError("Expected a PyPI package URL")
        mirror = "https://mirrors.aliyun.com/pypi" + source.path
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--max-time",
                "900",
                "--silent",
                "--show-error",
                "--output",
                str(partial),
                mirror,
            ],
            check=True,
        )
    if digest(partial) != item["sha256"]:
        raise ValueError(f"Wheel hash mismatch: {name}; partial retained, not published")
    partial.replace(target)
    print(f"downloaded and verified: {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-wheel", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads(args.manifest.read_text())["wheels"]
    names = [item["filename"] for item in items]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate wheel filenames")
    args.directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: fetch(item, args.directory, args.source_wheel), items))


if __name__ == "__main__":
    main()
