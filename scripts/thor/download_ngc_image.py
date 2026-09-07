"""Fetch a pinned public NGC manifest, reusing SHA-256 verified local blobs.

The output is a skopeo dir transport image; cache directories are read-only.
Anonymous pull tokens are kept in memory and never printed or persisted.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import requests


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_name(value):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("Only SHA-256 blob identifiers are accepted")
    return value.split(":")[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-digest", required=True)
    parser.add_argument("--repository", default="nvidia/pytorch")
    parser.add_argument("--cache", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Use between 1 and 16 workers")
    if not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+", args.repository):
        parser.error("Expected a plain NGC organization/repository")
    if digest(args.manifest) != checked_name(args.manifest_digest):
        parser.error("Manifest fingerprint mismatch")
    if args.output.resolve() in [path.resolve() for path in args.cache]:
        parser.error("Output must not be a source cache")
    manifest = json.loads(args.manifest.read_text())
    blobs = {item["digest"]: item for item in [manifest["config"], *manifest["layers"]]}
    args.output.mkdir(parents=True, exist_ok=True)
    published = args.output / "manifest.json"
    if published.exists() and digest(published) != checked_name(args.manifest_digest):
        parser.error("Output already belongs to a different image")
    auth = requests.get(
        "https://nvcr.io/proxy_auth",
        params={"scope": f"repository:{args.repository}:pull"},
        timeout=30,
    )
    auth.raise_for_status()
    token = auth.json()["token"]

    def fetch(item):
        name = checked_name(item["digest"])
        target = args.output / name
        if target.exists():
            if target.stat().st_size == item["size"] and digest(target) == name:
                print("EXISTING", name, item["size"], flush=True)
                return
            raise ValueError(f"Existing immutable blob is corrupt: {name}")
        for root in args.cache:
            source = root / name
            if source.is_file() and source.stat().st_size == item["size"] and digest(source) == name:
                shutil.copyfile(source, target)
                print("REUSED", name, item["size"], flush=True)
                return
        partial = target.with_suffix(".partial")
        if partial.exists() and partial.stat().st_size == item["size"] and digest(partial) == name:
            partial.rename(target)
            print("RESUMED_COMPLETE", name, item["size"], flush=True)
            return
        for attempt in range(4):
            try:
                start = partial.stat().st_size if partial.exists() else 0
                headers = {"Authorization": f"Bearer {token}"}
                if start:
                    headers["Range"] = f"bytes={start}-"
                # requests strips Authorization on cross-host redirects.
                with requests.get(
                    f"https://nvcr.io/v2/{args.repository}/blobs/{item['digest']}",
                    headers=headers,
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    response.raise_for_status()
                    resume = start and response.status_code == 206
                    if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise ValueError("Invalid resume range")
                    with partial.open("ab" if resume else "wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                if partial.stat().st_size != item["size"] or digest(partial) != name:
                    raise ValueError(f"Downloaded blob fingerprint mismatch: {name}")
                partial.rename(target)
                print("DOWNLOADED", name, item["size"], flush=True)
                return
            except requests.RequestException:
                # Do not log exceptions containing signed URLs or HTTP headers.
                if attempt == 3:
                    raise RuntimeError(f"Download failed after retries: {name}") from None
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, blobs.values()))
    # Publish the manifest only after every referenced blob has passed SHA-256.
    shutil.copyfile(args.manifest, args.output / "manifest.json")
    (args.output / "version").write_text("Directory Transport Version: 1.1\n")
    print("IMAGE_COMPLETE", args.manifest_digest, len(blobs), flush=True)


if __name__ == "__main__":
    main()
