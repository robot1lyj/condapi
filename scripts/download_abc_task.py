"""Download one ABC-130k task without re-encoding its mixed-task video shards.

Run on a CPU-only Slurm allocation. New directories only; retries preserve partial
files. Published manifests contain only episodes passing the declared structural
and conservative visual checks. This is not native-rate or Pantheon certification.
"""

import argparse
from collections import Counter
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request

CAMERAS = ("top", "left_wrist", "right_wrist")
HF_REVISIONS = {
    "train": "68651e4929d9fb00f798937b2d62617cab5c771d",
    "val": "46ca817c39d3df08115a75c31b3cd3867e3d875a",
}
POLICY = {
    "scope": "LeRobot 30fps structural and conservative visual QC; no task-success labels",
    "pantheon_coverage": "NOT_SCREENED: original UUIDs are absent from this LeRobot port",
    "native_rate_qc": "NOT_SCREENED: resampled timestamps cannot prove native cadence",
    "whole_episode_exclusion": True,
    "visual_resolution": [64, 64],
    "duplicate_luma_mad_below": 0.25,
    "duplicate_pair_fraction": "report only: periodic duplicates can be introduced by 30fps resampling",
    "frozen_run_seconds_above": 1.0,
    "extreme_exposure_fraction_above": 0.25,
    "low_contrast_fraction_above": 0.25,
    "nominal_gripper_out_of_range": "report only; do not clip or assume sensor corruption",
}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def api_url(repo, revision, path, *, tree=False):
    query = {"Revision": revision, "Root" if tree else "FilePath": path}
    return f"https://modelscope.cn/api/v1/datasets/{repo}/repo{'/tree' if tree else ''}?{urllib.parse.urlencode(query)}"


def get_tree(repo, revision, path):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(api_url(repo, revision, path, tree=True), timeout=90) as response:
                value = json.load(response)
            if value["Code"] != 200:
                raise RuntimeError("mirror tree API failed")
            return value["Data"]["Files"]
        except Exception:
            if attempt == 5:
                raise RuntimeError(f"Cannot list pinned source directory: {repo}/{path}") from None
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("Source listing retries exhausted")


class RangeReader(io.RawIOBase):
    """Bounded seekable HTTP reader. Reject ignored or malformed ranges."""

    def __init__(self, url, size, block_size=2 * 1024 * 1024):
        self.url, self.size, self.block_size = url, size, block_size
        self.active_url = url
        self.position, self.transferred = 0, 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if position < 0 or whence not in (0, 1, 2):
            raise ValueError("invalid seek")
        self.position = position
        return position

    def block(self, index):
        if index not in self.cache:
            start = index * self.block_size
            end = min(start + self.block_size, self.size) - 1
            for attempt in range(6):
                try:
                    request = urllib.request.Request(self.active_url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        expected = f"bytes {start}-{end}/{self.size}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise OSError("source does not honor the exact requested byte range")
                        # Reuse the temporary CDN URL in memory; refresh through the
                        # pinned API if it expires. Never log or persist signed URLs.
                        self.active_url = response.geturl()
                        value = response.read(end - start + 2)
                    if len(value) != end - start + 1:
                        raise OSError("short byte range")
                    self.transferred += len(value)
                    self.cache[index] = value
                    break
                except Exception:
                    self.active_url = self.url
                    if attempt == 5:
                        raise OSError("Source range unavailable after retries; not classified as bad data") from None
                    time.sleep(min(30, 2**attempt))
        self.cache.move_to_end(index)
        value = self.cache[index]
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return value

    def read(self, size=-1):
        size = max(0, self.size - self.position) if size < 0 else min(size, max(0, self.size - self.position))
        parts = []
        while size:
            index, offset = divmod(self.position, self.block_size)
            part = self.block(index)[offset : offset + size]
            parts.append(part)
            self.position += len(part)
            size -= len(part)
        return b"".join(parts)

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[: len(value)] = value
        return len(value)


def remote(repo, revision, record):
    return RangeReader(api_url(repo, revision, record["Path"]), record["Size"])


def download_metadata(repo, revision, record, output):
    if output.exists():
        if sha256(output) != record["Sha256"]:
            raise RuntimeError("Existing metadata hash mismatch; original retained")
        return
    reader = remote(repo, revision, record)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("wb") as stream:
        while value := reader.read(2 * 1024 * 1024):
            stream.write(value)
    if sha256(partial) != record["Sha256"]:
        raise RuntimeError("Downloaded metadata hash mismatch; not classified as episode damage")
    partial.replace(output)


def prepare(root, task):
    import pyarrow.parquet as pq  # noqa: PLC0415

    splits = {}
    for split in ("train", "val"):
        repo = f"lerobot/abc_130k_v3_{split}"
        listing = get_tree(repo, "master", "")
        # Pin each mirror to a file's full immutable commit, never download from master.
        revision = max((row for row in listing if row.get("Revision")), key=lambda row: row["CommittedDate"])[
            "Revision"
        ]
        info = get_tree(repo, revision, "meta")
        metadata = [row for row in info if row["Type"] == "blob"]
        for chunk in get_tree(repo, revision, "meta/episodes"):
            metadata.extend(get_tree(repo, revision, chunk["Path"]))
        episodes = []
        directory = root / "source-metadata" / split
        directory.mkdir(parents=True, exist_ok=True)
        for record in metadata:
            print(json.dumps({"phase": "metadata", "split": split, "path": record["Path"]}), flush=True)
            path = directory / record["Path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            download_metadata(repo, revision, record, path)
            if record["Path"].startswith("meta/episodes/"):
                schema = pq.read_schema(path)
                columns = [name for name in schema.names if not name.startswith("stats/")]
                episodes.extend(
                    row for row in pq.read_table(path, columns=columns).to_pylist() if row["tasks"] == [task]
                )
        info_json = json.loads((directory / "meta/info.json").read_text())
        if info_json["fps"] != 30 or not episodes:
            raise RuntimeError("Unexpected cadence or task absent from source")
        required = set()
        for row in episodes:
            required.add(f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet")
            for camera in CAMERAS:
                key = f"videos/observation.images.{camera}"
                required.add(f"{key}/chunk-{row[key + '/chunk_index']:03d}/file-{row[key + '/file_index']:03d}.mp4")
        files = {}
        for parent in sorted({str(Path(path).parent) for path in required}):
            files.update({row["Path"]: row for row in get_tree(repo, revision, parent) if row["Path"] in required})
        if set(files) != required:
            raise RuntimeError("Incomplete upstream inventory; do not misclassify as damaged episodes")
        splits[split] = {
            "source_repo": repo,
            "mirror_revision": revision,
            "published_hf_revision": HF_REVISIONS[split],
            "hf_file_hash_equivalence": "not independently checked; mirror SHA256 and revision bind this download",
            "metadata": metadata,
            "episodes": episodes,
            "files": files,
        }
        print(
            json.dumps(
                {
                    "phase": "planned",
                    "split": split,
                    "episodes": len(episodes),
                    "frames": sum(row["length"] for row in episodes),
                }
            ),
            flush=True,
        )
    plan = {"task": task, "policy": POLICY, "splits": splits}
    save_json(root / "plan.json", plan)
    return plan


def clip_video(reader, row, camera, output):
    import av  # noqa: PLC0415

    key = f"videos/observation.images.{camera}"
    start, end = row[key + "/from_timestamp"], row[key + "/to_timestamp"]
    if abs((end - start) * 30 - row["length"]) > 0.02:
        raise ValueError("source_video_interval_length_mismatch")
    partial = output.with_suffix(".partial.mp4")
    with av.open(reader) as source, av.open(str(partial), "w", format="mp4") as destination:
        stream = source.streams.video[0]
        stream.codec_context.thread_count = 1
        source.seek(int(start / stream.time_base), stream=stream, backward=True)
        target = destination.add_stream_from_template(stream, opaque=True)
        origin = None
        for packet in source.demux(stream):
            if packet.pts is None:
                continue
            timestamp = float(packet.pts * packet.time_base)
            if timestamp >= end - 1e-6:
                break
            if origin is None:
                if not packet.is_keyframe or timestamp > start + 1e-5:
                    raise ValueError("no_preceding_keyframe")
                origin = packet.pts
                offset = start - timestamp
            packet.pts -= origin
            packet.dts = None if packet.dts is None else packet.dts - origin
            packet.stream = target
            destination.mux(packet)
        if origin is None:
            raise ValueError("empty_video_interval")
    partial.replace(output)
    return offset


def audit_video(path, offset, length):
    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    count, duplicates, run, longest, exposure, contrast = 0, 0, 0, 0, 0, 0
    previous = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base)
            if timestamp < offset - 1e-5:
                continue
            if timestamp >= offset + length / 30 - 1e-5:
                continue
            if abs(timestamp - offset - count / 30) > 0.002:
                raise ValueError("video_timestamp_mismatch")
            pixels = frame.reformat(64, 64, format="gray").to_ndarray().astype(np.float32)
            if previous is not None:
                duplicate = float(np.abs(pixels - previous).mean()) < POLICY["duplicate_luma_mad_below"]
                duplicates += int(duplicate)
                run = run + 1 if duplicate else 0
                longest = max(longest, run)
            previous = pixels
            exposure += int(pixels.mean() < 5 or pixels.mean() > 250)
            contrast += int(pixels.std() < 3)
            count += 1
    if count != length:
        raise ValueError(f"decoded_frame_count:{count}!={length}")
    result = {
        "frames": count,
        "duplicate_fraction": duplicates / max(1, count - 1),
        "frozen_seconds": longest / 30,
        "exposure_fraction": exposure / count,
        "low_contrast_fraction": contrast / count,
    }
    flags = []
    for metric, threshold, name in (
        ("frozen_seconds", 1.0, "video_frozen_run"),
        ("exposure_fraction", 0.25, "video_extreme_exposure"),
        ("low_contrast_fraction", 0.25, "video_low_contrast"),
    ):
        if result[metric] > threshold:
            flags.append(name)
    return result, flags


def run(root, plan, limit=None):
    from audit_yam_subset import validate_rows  # noqa: PLC0415
    import av  # noqa: PLC0415
    import pyarrow.compute as pc  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    result_log = root / "audit.jsonl"
    previous = [json.loads(line) for line in result_log.read_text().splitlines()] if result_log.exists() else []
    done = {(r["split"], r["source_episode_index"]): r for r in previous}
    for split, source in plan["splits"].items():
        repo, revision = source["source_repo"], source["mirror_revision"]
        grouped = {}
        for row in source["episodes"]:
            path = f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
            grouped.setdefault(path, []).append(row)
        for path, rows in grouped.items():
            pending = [row for row in rows if (split, row["episode_index"]) not in done]
            if not pending:
                continue
            if shutil.disk_usage(root).free < 150 * 1024**3:
                raise OSError("Less than 150 GiB free; pause without excluding data")
            parquet = root / split / path
            parquet.parent.mkdir(parents=True, exist_ok=True)
            if not parquet.exists():
                with remote(repo, revision, source["files"][path]) as reader:
                    table = pq.read_table(reader)
                ids = [row["episode_index"] for row in rows]
                table = table.filter(pc.is_in(table["episode_index"], value_set=pc.cast(ids, "int64")))
                temporary = parquet.with_suffix(".partial.parquet")
                pq.write_table(table, temporary)
                temporary.replace(parquet)
            table = pq.read_table(parquet)
            data_sha = sha256(parquet)
            for row in pending:
                eid = row["episode_index"]
                record = {
                    "split": split,
                    "source_episode_index": eid,
                    "source_repo": repo,
                    "mirror_revision": revision,
                    "length": row["length"],
                    "fps": 30,
                    "task": plan["task"],
                    "data": str(parquet.relative_to(root)),
                    "data_sha256": data_sha,
                    "videos": {},
                    "issues": [],
                }
                subset = table.filter(pc.equal(table["episode_index"], eid))
                try:
                    validate_rows(subset, record)
                except ValueError as error:
                    record["issues"].append(str(error))
                for camera in CAMERAS if not record["issues"] else ():
                    key = f"videos/observation.images.{camera}"
                    source_path = f"{key}/chunk-{row[key + '/chunk_index']:03d}/file-{row[key + '/file_index']:03d}.mp4"
                    output = root / split / "videos" / camera / f"episode-{eid:06d}.mp4"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    receipt = output.with_suffix(".json")
                    try:
                        if receipt.exists() and output.exists():
                            clip = json.loads(receipt.read_text())
                            if sha256(output) != clip["sha256"]:
                                raise OSError("Existing extracted clip changed; stop and retain original")
                            offset = clip["offset_s"]
                        else:
                            with remote(repo, revision, source["files"][source_path]) as reader:
                                offset = clip_video(reader, row, camera, output)
                                transferred = reader.transferred
                            clip = {
                                "path": str(output.relative_to(root)),
                                "offset_s": offset,
                                "sha256": sha256(output),
                                "source_path": source_path,
                                "transferred_bytes": transferred,
                                "encoding": "original AV1 packets; no re-encode",
                            }
                            save_json(receipt, clip)
                        metrics, flags = audit_video(output, offset, row["length"])
                        record["videos"][camera] = dict(clip, metrics=metrics)
                        record["issues"].extend(f"{camera}:{flag}" for flag in flags)
                    except (ValueError, av.error.InvalidDataError) as error:
                        record["issues"].append(f"{camera}:{type(error).__name__}")
                record["status"] = "quarantined" if record["issues"] else "passed_declared_checks"
                with result_log.open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[split, eid] = record
                save_json(
                    root / "progress.json",
                    {
                        "status": "RUNNING",
                        "audited": len(done),
                        "expected": sum(len(s["episodes"]) for s in plan["splits"].values()),
                        "counts": dict(Counter(r["status"] for r in done.values())),
                        "last_episode": [split, eid],
                    },
                )
                print(
                    json.dumps(
                        {
                            "audited": len(done),
                            "split": split,
                            "episode": eid,
                            "status": record["status"],
                            "issues": record["issues"],
                        }
                    ),
                    flush=True,
                )
                if limit and len(done) >= limit:
                    return
    manifests = root / "manifests"
    manifests.mkdir(exist_ok=True)
    for split in plan["splits"]:
        accepted = [r for r in done.values() if r["split"] == split and r["status"] == "passed_declared_checks"]
        (manifests / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted))
    save_json(
        root / "completion.json",
        {
            "status": "DOWNLOAD_AND_DECLARED_QC_COMPLETE_NOT_TRAINING_READY",
            "policy": plan["policy"],
            "counts": dict(Counter(r["status"] for r in done.values())),
            "plan_sha256": sha256(root / "plan.json"),
            "audit_sha256": sha256(result_log),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", default="fold the paper box")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Requires CPU-only Slurm allocation, CUDA_VISIBLE_DEVICES='' ")
    args.root.mkdir(parents=True, exist_ok=True)
    import fcntl  # noqa: PLC0415

    with (args.root / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = args.root / "plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else prepare(args.root, args.task)
        if plan["task"] != args.task or plan["policy"] != POLICY:
            raise RuntimeError("Existing plan differs; use a new directory")
        if not args.prepare_only:
            run(args.root, plan, args.limit)


if __name__ == "__main__":
    main()
