"""Read-only audit of the uploaded ABC task subset; JSON report goes to stdout.

Inventory mode needs only Python's standard library. Full mode also needs
numpy, pyarrow and av. No source files are changed; missing/recent files are
pending upload, never automatically deleted or relabelled as corrupt.
"""

# Inventory runs on the login node without the training dependencies.
# ruff: noqa: PLC0415

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

CAMERAS = ("top", "left_wrist", "right_wrist")


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def validate_rows(table, episode):
    import numpy as np

    n = episode["length"]
    if table.num_rows != n:
        raise ValueError("row_count_mismatch")
    for key in ("observation.state", "action"):
        values = np.asarray(table[key].to_pylist())
        if values.shape != (n, 14) or not np.isfinite(values).all():
            raise ValueError(f"invalid_{key}")
    frames = np.asarray(table["frame_index"])
    if not np.array_equal(frames, np.arange(n)):
        raise ValueError("frame_index_not_contiguous")
    timestamps = np.asarray(table["timestamp"])
    if not np.allclose(timestamps, np.arange(n) / episode["fps"], atol=1e-3, rtol=0):
        raise ValueError("timestamp_mismatch")
    if not episode.get("task", "").strip():
        raise ValueError("missing_task")


def validate_video(path, length, fps):
    import av

    # Decode every frame to detect truncation; run full audits on compute nodes.
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - count / fps) > 1 / fps:
                raise ValueError("video_timestamp_mismatch")
            count += 1
    if count != length:
        raise ValueError(f"video_frame_count:{count}!={length}")


def audit(root, *, inventory_only=False, min_age=300, limit=None):
    root = Path(root).resolve()
    report = {
        "source": str(root),
        "observed_at": time.time(),
        "inventory_only": inventory_only,
        "trainable": False,
        "episodes": [],
        "manifests": {},
    }
    for split in ("train", "val"):
        manifest = root / "manifests" / f"{split}.jsonl"
        if not manifest.is_file():
            report["manifests"][split] = {"status": "pending_upload"}
            continue
        before_manifest = signature(manifest)
        content = manifest.read_bytes()
        try:
            episodes = [json.loads(line) for line in content.splitlines() if line.strip()]
        except (ValueError, UnicodeError):
            report["manifests"][split] = {"status": "pending_or_invalid_manifest"}
            continue
        report["manifests"][split] = {"sha256": hashlib.sha256(content).hexdigest(), "expected": len(episodes)}
        ids = Counter(e["source_episode_index"] for e in episodes)
        cached_path, cached_table = None, None
        for episode in episodes[:limit]:
            eid = episode["source_episode_index"]
            item = {
                "split": split,
                "source_episode_index": eid,
                "source_repo": episode["source_repo"],
                "source_revision": episode["source_revision"],
                "task": episode.get("task"),
                "length": episode["length"],
                "status": "inventory_ready",
            }
            report["episodes"].append(item)
            data = root / split / "data" / f"source-file-{episode['data']['file_index']:03d}.parquet"
            videos = [root / split / "videos" / cam / f"episode-{eid:06d}.mp4" for cam in CAMERAS]
            paths = [data, *videos]
            missing = [str(p.relative_to(root)) for p in paths if not p.is_file()]
            if missing:
                item.update(status="pending_upload", missing=missing)
                continue
            before = [signature(p) for p in paths]
            item["files"] = {str(p.relative_to(root)): list(s) for p, s in zip(paths, before, strict=True)}
            if any(size == 0 or time.time() - mtime / 1e9 < min_age for size, mtime in before):
                item["status"] = "pending_upload"
                continue
            if ids[eid] != 1:
                item.update(status="rejected", reason="duplicate_manifest_episode")
                continue
            if inventory_only:
                continue
            try:
                import pyarrow.compute as pc
                import pyarrow.parquet as pq

                cache_key = (data, before[0])
                if cached_path != cache_key:
                    cached_table = pq.read_table(data)
                    cached_path = cache_key
                table = cached_table.filter(pc.equal(cached_table["episode_index"], eid))
                validate_rows(table, episode)
                for video in videos:
                    validate_video(video, episode["length"], episode["fps"])
                item["status"] = "validated_structure"
            except (ValueError, OSError, KeyError, IndexError) as exc:
                item.update(status="rejected", reason=str(exc))
            if any(not p.is_file() or signature(p) != old for p, old in zip(paths, before, strict=True)):
                item.update(status="pending_upload", reason="changed_during_audit")
        if signature(manifest) != before_manifest:
            report["manifests"][split]["status"] = "changed_during_audit"
            for item in report["episodes"]:
                if item["split"] == split:
                    item["status"] = "pending_upload"
    report["counts"] = dict(Counter(item["status"] for item in report["episodes"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--min-age", type=float, default=300)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.min_age < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("min-age must be nonnegative and limit positive")
    print(json.dumps(audit(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
