"""Audit whole ABC episodes on CPU, exclude proven bad episodes, and replenish a duration subset.

Writes only a new report directory. Missing files, permission/I/O failures, and
changing sources stop the job; they are never treated as corrupt demonstrations.
No training or source mutation is performed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from select_lego_rtc_episodes import select_episodes
from yam_episode_exclusions import find_excluded_episodes


def audit_episode(task):
    # Optional data libraries are imported only inside compute-node workers.
    from audit_yam_subset import signature  # noqa: PLC0415
    from audit_yam_subset import validate_rows  # noqa: PLC0415
    from audit_yam_subset import validate_video  # noqa: PLC0415
    import av  # noqa: PLC0415
    import pyarrow.compute as pc  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    root, split, episode = task
    root = Path(root)
    source_id = episode["source_episode_index"]
    parquet = root / split / "data" / f"source-file-{episode['data']['file_index']:03d}.parquet"
    videos = [
        root / split / "videos" / camera / f"episode-{source_id:06d}.mp4"
        for camera in ("top", "left_wrist", "right_wrist")
    ]
    paths = [parquet, *videos]
    before = {str(path): signature(path) for path in paths}
    if any(size == 0 or time.time() - mtime / 1e9 < 300 for size, mtime in before.values()):
        raise RuntimeError("Empty/recent source may still be uploading; not classified as corruption")
    table = pq.read_table(parquet, columns=["observation.state", "action", "episode_index", "frame_index", "timestamp"])
    rows = table.filter(pc.equal(table["episode_index"], source_id))
    errors = []
    try:
        validate_rows(rows, episode)
    except ValueError as error:
        errors.append({"file": str(parquet.relative_to(root)), "error": str(error)})
    for video in videos:
        try:
            validate_video(video, episode["length"], episode["fps"], thread_count=1)
        except (ValueError, av.error.InvalidDataError) as error:
            with video.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            errors.append({"file": str(video.relative_to(root)), "error": str(error), "sha256": digest})
    if any(signature(path) != before[str(path)] for path in paths):
        raise RuntimeError("Source changed during audit; retry after upload is stable")
    return {
        "split": split,
        "source_episode_index": source_id,
        "length": episode["length"],
        "status": "excluded" if errors else "passed",
        "errors": errors,
        "source_signatures": {str(Path(path).relative_to(root)): list(sig) for path, sig in before.items()},
    }


def choose(rows, excluded, hours, seed):
    selected = select_episodes(
        [dict(row, episode_index=row["source_episode_index"]) for row in rows],
        fps=30,
        hours=hours,
        seed=seed,
        excluded_ids=excluded,
    )
    return [{key: value for key, value in row.items() if key != "episode_index"} for row in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Requires a CPU-only compute-node Slurm allocation")
    if not 1 <= args.workers <= int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Workers must fit the allocated CPU count")
    root = args.source.resolve()
    blobs = {split: (root / "manifests" / f"{split}.jsonl").read_bytes() for split in ("train", "val")}
    rows = {split: [json.loads(line) for line in blob.splitlines() if line.strip()] for split, blob in blobs.items()}
    qc_exclusions = {
        split: find_excluded_episodes(
            [dict(row, episode_index=row["source_episode_index"]) for row in members], split=split
        )
        for split, members in rows.items()
    }
    if {row["fps"] for split in rows.values() for row in split} != {30}:
        raise ValueError("Expected a consistent 30fps source")
    args.output.mkdir(parents=True, exist_ok=False)
    for split, blob in blobs.items():
        (args.output / f"source-{split}.jsonl").write_bytes(blob)
    (args.output / "contract.json").write_bytes(args.contract.read_bytes())
    results = {"train": {}, "val": {}}
    excluded = {row["source_episode_index"] for row in qc_exclusions["train"]}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / "audit.jsonl").open("x") as log:
        qc_excluded_val = {row["source_episode_index"] for row in qc_exclusions["val"]}
        pending_val = [row for row in rows["val"] if row["source_episode_index"] not in qc_excluded_val]
        while True:
            selected = choose(rows["train"], excluded, 50, 42)
            pending_train = [row for row in selected if row["source_episode_index"] not in results["train"]]
            pending = [(str(root), "val", row) for row in pending_val] + [
                (str(root), "train", row) for row in pending_train
            ]
            if not pending:
                break
            for result in pool.map(audit_episode, pending, chunksize=1):
                results[result["split"]][result["source_episode_index"]] = result
                if result["status"] == "excluded" and result["split"] == "train":
                    excluded.add(result["source_episode_index"])
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()
                done = sum(len(value) for value in results.values())
                if done % 20 == 0 or result["status"] == "excluded":
                    print(
                        json.dumps(
                            {
                                "audited": done,
                                "excluded_train": len(excluded),
                                "last": result["source_episode_index"],
                                "status": result["status"],
                                "seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
            pending_val = []
    for split, blob in blobs.items():
        if (root / "manifests" / f"{split}.jsonl").read_bytes() != blob:
            raise RuntimeError("Source manifest changed during audit")
    selected = choose(rows["train"], excluded, 50, 42)
    excluded_val = qc_excluded_val | {key for key, value in results["val"].items() if value["status"] == "excluded"}
    valid_val = [row for row in rows["val"] if row["source_episode_index"] not in excluded_val]
    if not valid_val:
        raise RuntimeError("No valid held-out episodes remain")
    frames = sum(row["length"] for row in selected)
    selection = {
        "schema_version": 1,
        "source_root": str(root),
        "source_manifest_sha256": {split: hashlib.sha256(blob).hexdigest() for split, blob in blobs.items()},
        "seed": 42,
        "fps": 30,
        "target_hours": 50,
        "frames": frames,
        "hours": frames / 30 / 3600,
        "source_train_episodes": len(rows["train"]),
        "source_train_frames": sum(row["length"] for row in rows["train"]),
        "val_episodes": len(valid_val),
        "val_frames": sum(row["length"] for row in valid_val),
        "excluded_val_source_episode_ids": sorted(excluded_val),
        "excluded_source_episode_ids": sorted(excluded),
        "known_qc_exclusions": {
            split: [{key: value for key, value in row.items() if key != "episode_index"} for row in members]
            for split, members in qc_exclusions.items()
        },
        "selected_episodes": selected,
        "review_status": "all selected numeric rows and all three videos fully audited; success not manually reviewed",
        "exclusion_policy": "whole episode excluded; original files retained; no frame trimming or repair; replenish in fixed seed42 order",
    }
    for name, value in (
        ("selection.json", selection),
        ("source-episodes.json", [row["source_episode_index"] for row in selected]),
        ("episodes.json", list(range(len(selected)))),
        (
            "excluded.json",
            [value for split in results.values() for value in split.values() if value["status"] == "excluded"],
        ),
    ):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "AUDIT_COMPLETE",
                "episodes": len(selected),
                "frames": frames,
                "hours": selection["hours"],
                "excluded": sorted(excluded),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
