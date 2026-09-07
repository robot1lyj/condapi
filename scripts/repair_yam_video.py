"""Replace one proven-corrupt source video, preserving it and rebasing its checkpoint.

Requires explicit expected hashes and a fully decoded replacement. This is an
operator-approved recovery, not a relaxation of normal resume identity checks.
Never run against an active conversion or an already published split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from lerobot.datasets.video_utils import get_video_info

if __package__:
    from .audit_yam_subset import signature
    from .audit_yam_subset import validate_video
    from .convert_yam_subset import digest
    from .convert_yam_subset import source_paths
    from .yam_conversion_resume import atomic_json
    from .yam_conversion_resume import conversion_lock
    from .yam_conversion_resume import sync_directory
    from .yam_conversion_resume import sync_file
else:
    from audit_yam_subset import signature
    from audit_yam_subset import validate_video
    from convert_yam_subset import digest
    from convert_yam_subset import source_paths
    from yam_conversion_resume import atomic_json
    from yam_conversion_resume import conversion_lock
    from yam_conversion_resume import sync_directory
    from yam_conversion_resume import sync_file


def repair(work, source, replacement, expected_bad_sha256, expected_good_sha256, evidence):
    work, source, replacement = (Path(p).absolute() for p in (work, source, replacement))
    if any(p.is_symlink() for p in (work, source, replacement)):
        raise ValueError("Symlinks are not recovery targets")
    work, source, replacement = (p.resolve() for p in (work, source, replacement))
    if not work.name.endswith(".incomplete") or work.with_suffix("").exists():
        raise ValueError("Only an unpublished split checkpoint can be repaired")
    if not evidence.strip() or source == replacement:
        raise ValueError("Separate replacement and upstream evidence are required")
    # Same locks as the batch runner and converter, acquired in the same order.
    batch_lock_work = work.parent.with_suffix("")
    with conversion_lock(batch_lock_work), conversion_lock(work):
        marker = work / "resume_identity.json"
        checkpoint = json.loads(marker.read_text())
        identity = checkpoint["identity"]
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Invalid checkpoint fingerprint")
        root = Path(identity["root"]).resolve()
        relative = str(source.relative_to(root))
        if relative not in identity["source_signatures"] or source.suffix != ".mp4":
            raise ValueError("Source is not a checkpoint video")
        manifest = root / "manifests" / f"{identity['split']}.jsonl"
        if digest(manifest) != identity["manifest_sha256"]:
            raise ValueError("Source manifest changed")
        for name, sig in identity["source_signatures"].items():
            if list(signature(root / name)) != sig:
                raise ValueError(f"Source signature changed: {name}")
        for name, sha in identity["parquet_sha256"].items():
            if digest(root / name) != sha:
                raise ValueError(f"Source parquet changed: {name}")
        episodes = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        matches = [e for e in episodes if source in source_paths(root, identity["split"], e)[1].values()]
        if len(matches) != 1 or matches[0]["source_episode_index"] not in identity["episode_ids"]:
            raise ValueError("Source must belong to exactly one selected episode")
        episode = matches[0]
        if digest(source) != expected_bad_sha256 or digest(replacement) != expected_good_sha256:
            raise ValueError("Explicit source/replacement SHA-256 mismatch")
        try:
            validate_video(source, episode["length"], episode["fps"])
        except ValueError as error:
            if not str(error).startswith(("video_frame_count:", "video_timestamp_mismatch")):
                raise
            failure = str(error)
        else:
            raise ValueError("Source passes validation; refusing replacement")
        validate_video(replacement, episode["length"], episode["fps"])
        if get_video_info(source) != get_video_info(replacement):
            raise ValueError("Replacement stream format differs from source")
        nonce = time.time_ns()
        backup = source.with_name(source.name + f".corrupt-{nonce}")
        temporary = source.with_name(source.name + f".recovery-{nonce}")
        record_path = work / "resume_records" / f"source-repair-{nonce}.json"
        record = {
            "source": relative,
            "source_episode_index": episode["source_episode_index"],
            "old_sha256": expected_bad_sha256,
            "new_sha256": expected_good_sha256,
            "backup": str(backup),
            "replacement": str(replacement),
            "evidence": evidence,
            "failure": failure,
            "old_fingerprint": fingerprint,
            "status": "prepared",
        }
        atomic_json(record_path, record)
        atomic_json(marker.with_name(f"resume_identity.before-repair-{nonce}.json"), checkpoint)
        with replacement.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        sync_file(temporary)
        if digest(temporary) != expected_good_sha256 or digest(source) != expected_bad_sha256:
            raise ValueError("Video changed during recovery; preserved prepared record")
        source.rename(backup)
        temporary.rename(source)
        sync_directory(source.parent)
        identity["source_signatures"][relative] = list(signature(source))
        checkpoint["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        record.update(status="complete", new_fingerprint=checkpoint["fingerprint"])
        checkpoint.setdefault("source_repairs", []).append(record)
        atomic_json(marker, checkpoint)
        atomic_json(record_path, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "source", "replacement"):
        parser.add_argument(name, type=Path)
    for name in ("expected-bad-sha256", "expected-good-sha256", "evidence"):
        parser.add_argument(f"--{name}", required=True)
    print(json.dumps(repair(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
