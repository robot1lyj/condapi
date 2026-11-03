"""Prepare immutable ABC T-shirt parent episodes for later FineART annotation.

CPU file preparation only: numpy, pyarrow and PyAV; no model/training imports.
All artifacts belong to a new --root. Resume only with the exact same inputs.
Visual metrics request REVIEW; only structural errors quarantine whole episodes.
No output is certified task-successful or training-ready.
"""

# ruff: noqa: PLC0415
import argparse
from collections import Counter
from collections import OrderedDict
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid

REPO = "lerobot/abc_130k_v3_train"
HF_REVISION = "68651e4929d9fb00f798937b2d62617cab5c771d"
MIRROR_REVISION = "5b1e1b0e4384922c118b0fd4a68d803feea37d73"
CAMERAS = ("top", "left_wrist", "right_wrist")
POOLS = ("train", "dev", "reserve")
TASK = "fold and stack the t-shirts"
FORMAT_VERSION = 1


class InfrastructureError(RuntimeError):
    """Stop/retry: transport, inventory, capacity or existing artifact conflict."""


class StructuralError(ValueError):
    """A verified source episode violates the declared structural contract."""


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def source_paths(row):
    result = {"data": f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"}
    for camera in CAMERAS:
        key = f"videos/observation.images.{camera}"
        result[camera] = f"{key}/chunk-{row[key + '/chunk_index']:03d}/file-{row[key + '/file_index']:03d}.mp4"
    return result


def make_plan(manifests, source_metadata, pilot_path):
    inventory = load(source_metadata)
    if inventory.get("repo") != REPO or inventory.get("revision") != HF_REVISION:
        raise InfrastructureError("Source inventory is not the pinned HF repository/revision")
    files = {x["path"]: x for x in inventory["files"]}
    if len(files) != len(inventory["files"]):
        raise InfrastructureError("Duplicate paths in HF inventory")
    episodes, seen = [], set()
    inputs = {"source_metadata": digest(source_metadata), "pilots": digest(pilot_path)}
    pilot_ids = [int(x["source_episode_index"]) for x in load(pilot_path)["pilots"]]
    if len(set(pilot_ids)) != len(pilot_ids):
        raise InfrastructureError("Duplicate pilot episode IDs")
    for pool in POOLS:
        path = manifests[pool]
        inputs[pool] = digest(path)
        for row in load(path):
            eid = int(row["episode_index"])
            if eid in seen:
                raise InfrastructureError("Candidate pools overlap by source episode ID")
            seen.add(eid)
            if row.get("split") != "train" or row.get("tasks") != [TASK] or row["length"] <= 0:
                raise InfrastructureError("Unexpected split/task/length in candidate input")
            for path in source_paths(row).values():
                item = files.get(path)
                lfs = item.get("lfs", {}) if item else {}
                if not item or item.get("type") != "file" or item["size"] != lfs.get("size"):
                    raise InfrastructureError("Missing or inconsistent pinned HF LFS metadata: " + path)
                sha = lfs.get("oid", "")
                if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                    raise InfrastructureError("Missing valid HF LFS SHA256: " + path)
            episodes.append({"pool": pool, "source": row})
    if not set(pilot_ids).issubset({e["source"]["episode_index"] for e in episodes if e["pool"] == "train"}):
        raise InfrastructureError("Pilot IDs must all belong to the train pool")
    return {
        "format_version": FORMAT_VERSION,
        "repo": REPO,
        "hf_revision": HF_REVISION,
        "mirror_revision": MIRROR_REVISION,
        "input_sha256": inputs,
        "episodes": episodes,
        "pilot_ids": pilot_ids,
        "files": files,
        "contract": {
            "fps": 30,
            "state_action": "original 14D absolute; all columns/schema preserved",
            "video": "original AV1 packet remux, 224x224; per-frame count/timestamp QC",
            "task_success": "NOT_REVIEWED",
            "training_ready": False,
            "visual_flags": "REVIEW only; stillness is expected in cloth manipulation",
            "quarantine": "whole episode only for structural violations; preserve every artifact",
        },
    }


def api_url(path, *, tree=False, page=1):
    query = {"Revision": MIRROR_REVISION, "Root" if tree else "FilePath": path}
    if tree:
        query.update(PageNumber=page, PageSize=1000)
    return f"https://modelscope.cn/api/v1/datasets/{REPO}/repo{'/tree' if tree else ''}?{urllib.parse.urlencode(query)}"


def request_bytes(url, *, headers=None, maximum, timeout, attempts):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = response.read(maximum + 1)
                if len(value) > maximum:
                    raise InfrastructureError("Response exceeded the bounded request size")
                return response.status, response.headers, response.geturl(), value
        except (OSError, InfrastructureError) as error:
            if attempt + 1 == attempts:
                raise InfrastructureError("Source unavailable; retry without marking an episode corrupt") from error
            time.sleep(min(4, 2**attempt))
    raise AssertionError("unreachable")


class MirrorInventory:
    def __init__(self, root, plan, timeout=20, attempts=3):
        self.root, self.plan = root, plan
        self.timeout, self.attempts = timeout, attempts
        self.directories = {}

    def verify(self, path):
        parent = str(Path(path).parent)
        if parent not in self.directories:
            cache = self.root / "source-inventory" / (hashlib.sha256(parent.encode()).hexdigest() + ".json")
            if cache.exists():
                saved = load(cache)
                if saved.get("revision") != MIRROR_REVISION or saved.get("directory") != parent:
                    raise InfrastructureError("Mirror listing cache identity mismatch")
                records = saved["files"]
            else:
                records, seen, page = [], set(), 1
                while True:
                    _, _, _, raw = request_bytes(
                        api_url(parent, tree=True, page=page),
                        maximum=4 * 1024**2,
                        timeout=self.timeout,
                        attempts=self.attempts,
                    )
                    value = json.loads(raw)
                    if value.get("Code") != 200:
                        raise InfrastructureError("Pinned mirror directory listing failed")
                    batch, total = value["Data"]["Files"], value["Data"]["TotalCount"]
                    for row in batch:
                        if row["Path"] in seen:
                            raise InfrastructureError("Duplicate mirror page; incomplete inventory")
                        seen.add(row["Path"])
                        records.append(row)
                    if len(records) == total:
                        break
                    if not batch or len(records) > total:
                        raise InfrastructureError("Incomplete mirror directory listing")
                    page += 1
                save_json(cache, {"revision": MIRROR_REVISION, "directory": parent, "files": records})
            self.directories[parent] = {x["Path"]: x for x in records}
        row = self.directories[parent].get(path)
        hf = self.plan["files"][path]
        if (
            not row
            or row.get("Type") != "blob"
            or row.get("Size") != hf["size"]
            or row.get("Sha256") != hf["lfs"]["oid"]
        ):
            raise InfrastructureError("Mirror Size/Sha256 differs from pinned HF LFS metadata: " + path)
        return {"path": path, "size": hf["size"], "sha256": hf["lfs"]["oid"]}


class RangeReader(io.RawIOBase):
    """Exact HTTP ranges, persistent per-block SHA receipts, bounded memory cache."""

    def __init__(self, root, identity, *, timeout=20, attempts=3, block_size=2 * 1024**2):
        self.identity, self.size = identity, identity["size"]
        self.position, self.transferred, self.failure = 0, 0, None
        self.timeout, self.attempts, self.block_size = timeout, attempts, block_size
        self.url = api_url(identity["path"])
        self.active_url = self.url  # Signed redirects remain in memory only.
        self.cache = OrderedDict()
        self.directory = root / "source-blocks" / identity["sha256"]
        self.directory.mkdir(parents=True, exist_ok=True)
        receipt = self.directory / "identity.json"
        expected = dict(identity, block_size=block_size)
        if receipt.exists() and load(receipt) != expected:
            raise InfrastructureError("Block cache identity differs; retain existing files")
        if not receipt.exists():
            save_json(receipt, expected)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek mode")
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def block(self, index):
        if index not in self.cache:
            start, end = index * self.block_size, min((index + 1) * self.block_size, self.size) - 1
            path = self.directory / f"{index:08d}.block"
            receipt = path.with_suffix(".json")
            try:
                if path.exists() or receipt.exists():
                    if not receipt.exists():
                        raise InfrastructureError("Unreceipted source block; original retained")
                    saved = load(receipt)
                    if not path.exists():
                        temporary = self.directory / saved.get("partial", "missing")
                        if not temporary.is_file() or digest(temporary) != saved.get("sha256"):
                            raise InfrastructureError("Incomplete block commit; retained for inspection")
                        temporary.replace(path)
                    if saved.get("bytes") != end - start + 1 or digest(path) != saved.get("sha256"):
                        raise InfrastructureError("Cached source block SHA mismatch; original retained")
                    value = path.read_bytes()
                else:
                    status, headers, active, value = request_bytes(
                        self.active_url,
                        headers={"Range": f"bytes={start}-{end}"},
                        maximum=end - start + 1,
                        timeout=self.timeout,
                        attempts=self.attempts,
                    )
                    if (
                        status != 206
                        or headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                        or len(value) != end - start + 1
                    ):
                        raise InfrastructureError("Source ignored exact byte range; no damage classification")
                    self.active_url = active
                    temporary = path.with_suffix(".partial-" + uuid.uuid4().hex)
                    with temporary.open("xb") as stream:
                        stream.write(value)
                    # Receipt-first allows an interrupted rename to be recovered from its named partial.
                    save_json(
                        receipt,
                        {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest(), "partial": temporary.name},
                    )
                    temporary.replace(path)
                    self.transferred += len(value)
                self.cache[index] = value
            except (OSError, InfrastructureError) as error:
                self.failure = error
                raise InfrastructureError("Range/cache I/O failure; do not quarantine episode") from error
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


def retain_partial(path):
    if path.exists():
        path.rename(path.with_name(path.name + ".retained-" + uuid.uuid4().hex))


def artifact(root, path):
    return {"path": str(path.relative_to(root)), "sha256": digest(path), "bytes": path.stat().st_size}


def verify_artifacts(root, record):
    for item in record.get("artifacts", []):
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise InfrastructureError("Previously committed artifact changed; original retained: " + str(path))


def recover_commit(root, output, receipt, item):
    """Recover receipt-before-rename after interruption without overwriting files."""
    if not output.exists():
        partial = root / item.get("partial_path", "missing")
        if not partial.is_file() or digest(partial) != item["sha256"]:
            raise InfrastructureError("Incomplete artifact commit; retain and inspect: " + str(receipt))
        partial.replace(output)
    verify_artifacts(root, {"artifacts": [item]})


def source_parquet(root, identity, reader_options):
    output = root / "source-parquet" / (identity["sha256"] + ".parquet")
    if output.exists():
        if output.stat().st_size != identity["size"] or digest(output) != identity["sha256"]:
            raise InfrastructureError("Cached source Parquet SHA mismatch; original retained")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial")
    retain_partial(partial)
    with RangeReader(root, identity, **reader_options) as reader, partial.open("xb") as stream:
        while value := reader.read(2 * 1024**2):
            stream.write(value)
    if digest(partial) != identity["sha256"]:
        raise InfrastructureError("Source Parquet payload differs from HF/mirror SHA; retain partial and stop")
    partial.replace(output)
    return output


def validate_rows(table, row):
    import numpy as np

    n = row["length"]
    required = {"episode_index", "observation.state", "action", "frame_index", "timestamp"}
    if not required.issubset(table.column_names) or table.num_rows != n:
        raise StructuralError("missing_columns_or_episode_row_count_mismatch")
    if set(table["episode_index"].to_pylist()) != {row["episode_index"]}:
        raise StructuralError("source_episode_id_mismatch")
    for key in ("observation.state", "action"):
        try:
            values = np.asarray(table[key].to_pylist())
            valid = values.shape == (n, 14) and np.isfinite(values).all()
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise StructuralError("invalid_14D_or_nonfinite_" + key)
    if table["frame_index"].to_pylist() != list(range(n)):
        raise StructuralError("frame_index_not_contiguous")
    if not np.allclose(np.asarray(table["timestamp"]), np.arange(n) / 30, atol=1e-3, rtol=0):
        raise StructuralError("timestamp_not_30fps_episode_relative")


def remux_video(reader, row, camera, partial, *, expected_codec="av1"):
    import av

    key = "videos/observation.images." + camera
    start, end = row[key + "/from_timestamp"], row[key + "/to_timestamp"]
    if start < 0 or abs((end - start) * 30 - row["length"]) > 0.02:
        raise StructuralError("source_video_interval_length_mismatch")
    try:
        with av.open(reader) as source, av.open(str(partial), "w", format="mp4") as output:
            if len(source.streams.video) != 1:
                raise StructuralError("expected_one_video_stream")
            stream = source.streams.video[0]
            stream.codec_context.thread_count = 1
            if stream.codec_context.codec.id != av.Codec(expected_codec, "r").id or (stream.width, stream.height) != (
                224,
                224,
            ):
                raise StructuralError("unexpected_codec_or_resolution")
            source.seek(int(start / stream.time_base), stream=stream, backward=True)
            target = output.add_stream_from_template(stream, opaque=True)
            origin, offset = None, None
            for packet in source.demux(stream):
                if packet.pts is None:
                    continue
                timestamp = float(packet.pts * packet.time_base)
                if timestamp >= end - 1e-6:
                    break
                if origin is None:
                    if not packet.is_keyframe or timestamp > start + 1e-5:
                        raise StructuralError("no_preceding_keyframe")
                    origin, offset = packet.pts, start - timestamp
                packet.pts -= origin
                packet.dts = None if packet.dts is None else packet.dts - origin
                packet.stream = target
                output.mux(packet)
            if origin is None:
                raise StructuralError("empty_video_interval")
        return offset
    except av.error.InvalidDataError as error:
        if getattr(reader, "failure", None):
            raise InfrastructureError("Transport error surfaced through video demux") from error
        raise StructuralError("video_demux_invalid_data") from error


def audit_video(path, offset, length):
    import av
    import numpy as np

    count, decoded, duplicates, run, longest, exposure, contrast = 0, 0, 0, 0, 0, 0, 0
    previous = None
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            previous_time = None
            for frame in container.decode(stream):
                decoded += 1
                if frame.pts is None:
                    raise StructuralError("video_missing_pts")
                timestamp = float(frame.pts * frame.time_base)
                if previous_time is not None and timestamp <= previous_time:
                    raise StructuralError("video_nonmonotonic_pts")
                previous_time = timestamp
                if timestamp < offset - 1e-5:
                    continue
                if timestamp >= offset + length / 30 - 1e-5:
                    raise StructuralError("unexpected_extra_video_frames")
                if abs(timestamp - offset - count / 30) > 0.002:
                    raise StructuralError("video_timestamp_mismatch")
                pixels = frame.reformat(64, 64, format="gray").to_ndarray().astype(np.float32)
                if previous is not None:
                    duplicate = float(np.abs(pixels - previous).mean()) < 0.25
                    duplicates += int(duplicate)
                    run = run + 1 if duplicate else 0
                    longest = max(longest, run)
                previous = pixels
                exposure += int(pixels.mean() < 5 or pixels.mean() > 250)
                contrast += int(pixels.std() < 3)
                count += 1
    except av.error.InvalidDataError as error:
        raise StructuralError("video_decode_invalid_data") from error
    if count != length:
        raise StructuralError(f"decoded_frame_count:{count}!={length}")
    metrics = {
        "frames": count,
        "total_decoded_with_preroll": decoded,
        "duplicate_fraction": duplicates / max(1, count - 1),
        "frozen_seconds": longest / 30,
        "exposure_fraction": exposure / count,
        "low_contrast_fraction": contrast / count,
    }
    flags = [
        name
        for key, threshold, name in [
            ("frozen_seconds", 1, "REVIEW_static_run"),
            ("exposure_fraction", 0.25, "REVIEW_extreme_exposure"),
            ("low_contrast_fraction", 0.25, "REVIEW_low_contrast"),
        ]
        if metrics[key] > threshold
    ]
    return metrics, flags


def prepare_episode(root, entry, subset, inventory, reader_options):
    import pyarrow.parquet as pq

    row, pool = entry["source"], entry["pool"]
    eid = row["episode_index"]
    directory = root / "raw" / pool / f"episode-{eid:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "pool": pool,
        "source_episode_index": eid,
        "source_repo": REPO,
        "source_revision": HF_REVISION,
        "mirror_revision": MIRROR_REVISION,
        "task": TASK,
        "fps": 30,
        "length": row["length"],
        "duration_s": row["length"] / 30,
        "state_action_semantics": "original_absolute_14D",
        "task_success": "NOT_REVIEWED",
        "training_ready": False,
        "videos": {},
        "review_flags": [],
        "structural_errors": [],
        "artifacts": [],
    }
    metadata_path = directory / "source-episode.json"
    if metadata_path.exists() and load(metadata_path) != row:
        raise InfrastructureError("Existing source-episode metadata differs")
    if not metadata_path.exists():
        save_json(metadata_path, row)
    try:
        validate_rows(subset, row)
        data = directory / "data.parquet"
        receipt = directory / "data.receipt.json"
        if data.exists() or receipt.exists():
            if not receipt.exists():
                raise InfrastructureError("Incomplete episode Parquet commit; retain and inspect")
            recover_commit(root, data, receipt, load(receipt))
        else:
            partial = directory / "data.partial.parquet"
            retain_partial(partial)
            pq.write_table(subset, partial)  # All columns + schema; no delta, padding, clipping or normalization.
            data_artifact = artifact(root, partial)
            data_artifact["path"] = str(data.relative_to(root))
            data_artifact["partial_path"] = str(partial.relative_to(root))
            save_json(receipt, data_artifact)
            partial.replace(data)
        record["data"] = artifact(root, data)
        for camera, source_path in source_paths(row).items():
            if camera == "data":
                continue
            identity = inventory.verify(source_path)  # Required BEFORE even one source payload range.
            output = directory / (camera + ".mp4")
            receipt = directory / (camera + ".receipt.json")
            if output.exists() or receipt.exists():
                if not receipt.exists():
                    raise InfrastructureError("Incomplete clip commit; retain and inspect")
                clip = load(receipt)
                recover_commit(root, output, receipt, clip["artifact"])
            else:
                partial = directory / (camera + ".partial.mp4")
                retain_partial(partial)
                with RangeReader(root, identity, **reader_options) as reader:
                    offset = remux_video(reader, row, camera, partial)
                    transferred = reader.transferred
                metrics, flags = audit_video(partial, offset, row["length"])
                item = artifact(root, partial)
                item["path"] = str(output.relative_to(root))
                item["partial_path"] = str(partial.relative_to(root))
                clip = {
                    "artifact": item,
                    "offset_s": offset,
                    "source_path": source_path,
                    "source_sha256": identity["sha256"],
                    "source_size": identity["size"],
                    "encoding": "original AV1 packets; no re-encode",
                    "metrics": metrics,
                    "review_flags": flags,
                    "new_transferred_bytes": transferred,
                }
                save_json(receipt, clip)
                partial.replace(output)
            record["videos"][camera] = clip
            record["review_flags"].extend(camera + ":" + flag for flag in clip["review_flags"])
        record["status"] = "PREPARED_REVIEW_PENDING"
    except StructuralError as error:
        record["structural_errors"].append(str(error))
        record["status"] = "QUARANTINED_STRUCTURE"
        save_json(
            root / "quarantine" / pool / f"episode-{eid:06d}.json",
            {"reason": str(error), "raw_directory": str(directory.relative_to(root)), "preserve": True},
        )
    record["artifacts"] = [
        artifact(root, p) for p in sorted(directory.iterdir()) if p.is_file() and not p.name.startswith("record.")
    ]
    save_json(directory / "record.json", record)
    return record


def publish(root, plan, done, selected_ids):
    records = sorted(done.values(), key=lambda r: (POOLS.index(r["pool"]), r["source_episode_index"]))
    for pool in POOLS:
        target = root / "manifests" / (pool + ".jsonl")
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_suffix(".partial.jsonl")
        temporary.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in records
                if r["pool"] == pool and r["status"] == "PREPARED_REVIEW_PENDING"
            )
        )
        temporary.replace(target)
    # Regenerated from atomic episode records; no truncated append-only line on interruption.
    audit = root / "audit.jsonl"
    temporary = root / "audit.partial.jsonl"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    temporary.replace(audit)
    completed = set(done)
    state = {
        "status": "PREPARATION_COMPLETE_NOT_TRAINING_READY"
        if len(done) == len(plan["episodes"])
        else "PARTIAL_PREPARATION_NOT_TRAINING_READY",
        "training_ready": False,
        "task_success": "NOT_REVIEWED",
        "processed": len(done),
        "expected": len(plan["episodes"]),
        "selected_remaining": len(set(selected_ids) - completed),
        "counts": dict(Counter(r["status"] for r in records)),
        "plan_sha256": digest(root / "plan.json"),
        "audit_sha256": digest(audit),
        "updated_unix_s": time.time(),
    }
    save_json(root / "progress.json", state)
    save_json(root / "completion.json", state)
    return state


def run(root, plan, *, pilot_only=False, limit=None, min_free_gb=5, timeout=20, attempts=3):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    reader_options = {"timeout": timeout, "attempts": attempts}
    inventory = MirrorInventory(root, plan, timeout, attempts)
    rank = {eid: i for i, eid in enumerate(plan["pilot_ids"])}
    selected = [e for e in plan["episodes"] if not pilot_only or e["source"]["episode_index"] in rank]
    selected.sort(
        key=lambda e: (
            rank.get(e["source"]["episode_index"], 10**9),
            POOLS.index(e["pool"]),
            e["source"]["episode_index"],
        )
    )
    done = {}
    for entry in plan["episodes"]:
        eid, pool = entry["source"]["episode_index"], entry["pool"]
        path = root / "raw" / pool / f"episode-{eid:06d}" / "record.json"
        if path.exists():
            record = load(path)
            if record["source_episode_index"] != eid or record["pool"] != pool:
                raise InfrastructureError("Episode record identity mismatch")
            verify_artifacts(root, record)
            done[eid] = record
    ids = [e["source"]["episode_index"] for e in selected]
    publish(root, plan, done, ids)
    grouped = OrderedDict()
    for entry in selected:
        if entry["source"]["episode_index"] not in done:
            grouped.setdefault(source_paths(entry["source"])["data"], []).append(entry)
    processed = 0
    for source_path, entries in grouped.items():
        if limit is not None and processed >= limit:
            break
        if shutil.disk_usage(root).free < min_free_gb * 1e9:
            raise InfrastructureError("Insufficient free space; resume without quarantining data")
        identity = inventory.verify(source_path)
        cached = source_parquet(root, identity, reader_options)
        try:
            table = pq.read_table(cached, use_threads=False)
        except Exception as error:
            # Source file is fully SHA-verified; unreadable group remains pending to inspect, not auto drop all children.
            raise InfrastructureError("Verified source Parquet unreadable; retain and inspect") from error
        for entry in entries:
            if limit is not None and processed >= limit:
                break
            row = entry["source"]
            if "episode_index" not in table.column_names:
                subset = table.slice(0, 0)
            else:
                subset = table.filter(pc.equal(table["episode_index"], row["episode_index"]))
            record = prepare_episode(root, entry, subset, inventory, reader_options)
            done[row["episode_index"]] = record
            processed += 1
            state = publish(root, plan, done, ids)
            print(
                json.dumps(
                    {
                        "new_processed": processed,
                        "source_episode_index": row["episode_index"],
                        "pool": entry["pool"],
                        "episode_status": record["status"],
                        "review_flags": record["review_flags"],
                        "processed_total": state["processed"],
                    }
                ),
                flush=True,
            )
        del table
    return publish(root, plan, done, ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    for pool in POOLS:
        parser.add_argument("--" + pool + "-manifest", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--pilots", type=Path, required=True)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum NEW episodes processed this invocation, not lifetime total")
    parser.add_argument("--min-free-gb", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args(argv)
    if (
        (args.limit is not None and args.limit < 1)
        or args.min_free_gb < 1
        or not 1 <= args.timeout <= 60
        or not 1 <= args.attempts <= 5
    ):
        parser.error("Positive limit, >=1GB free, timeout1..60s and attempts1..5 required")
    root = args.root.expanduser().resolve()
    manifests = {pool: getattr(args, pool + "_manifest") for pool in POOLS}
    plan = make_plan(manifests, args.source_metadata, args.pilots)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "ownership.json").exists() and {p.name for p in root.iterdir()} - {".prepare.lock"}:
        raise InfrastructureError("Output root is nonempty and unowned; use a NEW directory")
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker, plan_path = root / "ownership.json", root / "plan.json"
        if not marker.exists():
            if {p.name for p in root.iterdir()} != {".prepare.lock"}:
                raise InfrastructureError("Output root is nonempty and not owned by this preparer; use a NEW directory")
            save_json(marker, {"tool": "prepare_tshirt_fineart", "format_version": FORMAT_VERSION})
        if load(marker) != {"tool": "prepare_tshirt_fineart", "format_version": FORMAT_VERSION}:
            raise InfrastructureError("Output directory ownership mismatch")
        if plan_path.exists() and load(plan_path) != plan:
            raise InfrastructureError("Existing plan/inputs differ; use a NEW root")
        if not plan_path.exists():
            save_json(plan_path, plan)
        try:
            result = run(
                root,
                plan,
                pilot_only=args.pilot_only,
                limit=args.limit,
                min_free_gb=args.min_free_gb,
                timeout=args.timeout,
                attempts=args.attempts,
            )
        except Exception as error:
            save_json(
                root / "last-stop.json",
                {
                    "status": "STOPPED_RETRY_OR_INSPECT_NOT_EPISODE_DAMAGE",
                    "type": type(error).__name__,
                    "message": str(error),
                    "updated_unix_s": time.time(),
                },
            )
            raise
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (InfrastructureError, OSError) as exc:
        print(f"Preparation stopped (not damage classification): {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
