"""Create synchronized annotation-only windows from audited parent episodes.

Remux the original packets; never re-encode or alter the parent episode. FLORA
outputs stay window-local and require a reviewed parent-timeline conversion.
"""

# PyAV and native Flora are optional until a preparation operation is requested.
# ruff: noqa: PLC0415

import argparse
import fcntl
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import uuid

CAMERAS = ("top", "left_wrist", "right_wrist")
FPS = 30


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def intervals(length):
    if length <= 3600:
        return [(0, length, 0, length)]
    return [(a, min(a + 3300, length), max(0, a - 150), min(length, a + 3450)) for a in range(0, length, 3300)]


def remux(source, output, start, length):
    import av

    if output.exists():
        raise FileExistsError(output)
    partial = output.with_suffix(".partial.mp4")
    if partial.exists():
        partial.rename(partial.with_name(partial.name + ".retained-" + uuid.uuid4().hex))
    with (
        av.open(str(source)) as src,
        av.open(
            str(partial),
            "w",
            format="mp4",
            options={"avoid_negative_ts": "disabled", "use_editlist": "1"},
        ) as dst,
    ):
        stream = src.streams.video[0]
        origin = round(start / stream.time_base)
        # Parent offsets originate in floating-point shard timestamps. Snap both
        # ends to the same source time base: a +1e-13s residual must not include
        # the frame exactly at the nominal exclusive end.
        end = origin * stream.time_base + Fraction(length, FPS)
        src.seek(origin, stream=stream, backward=True)
        target = dst.add_stream_from_template(stream, opaque=True)
        copied = 0
        for packet in src.demux(stream):
            if packet.pts is None:
                continue
            if packet.pts * packet.time_base >= end:
                break
            if copied == 0 and (not packet.is_keyframe or packet.pts > origin):
                raise ValueError("No preceding keyframe; retain source and investigate")
            packet.pts -= origin
            if packet.dts is not None:
                packet.dts -= origin
            packet.stream = target
            dst.mux(packet)
            copied += 1
        if copied == 0:
            raise ValueError("No packets in annotation window")
    # Validate the edit list actually hides preroll, rather than assuming it does.
    n = 0
    with av.open(str(partial), options={"err_detect": "explode"}) as check:
        stream = check.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in check.decode(stream):
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - n / FPS) > 1e-5:
                raise ValueError("Remuxed window timestamp/preroll mismatch; no re-encoding attempted")
            n += 1
    if n != length:
        raise ValueError(f"Remuxed window frame count {n} != {length}; retain partial")
    if output.exists():
        raise FileExistsError(output)
    partial.replace(output)


def prepare(root, flora_root, output, pools, episode_ids, limit):
    sys.path.insert(0, str(flora_root / "captioner"))
    from batch_utils.api import DataEnginePipeline

    if output.exists():
        binding = output / "binding.json"
        if not binding.exists():
            raise RuntimeError("Existing output has no binding; choose a new directory")
    else:
        output.mkdir(parents=True)
    binding = {
        "root": str(root.resolve()),
        "fps": FPS,
        "core_frames": 3300,
        "context_frames": 150,
        "pools": pools,
        "episode_ids": sorted(episode_ids),
        "input_audit_sha256": digest(root / "audit.jsonl"),
        "encoding": "original AV1 packets, annotation-only MP4 edit lists",
    }
    binding_path = output / "binding.json"
    if binding_path.exists():
        previous = json.loads(binding_path.read_text())
        # The audit is append-only while the download runs. Other fields bind the view.
        for key in binding.keys() - {"input_audit_sha256"}:
            if previous[key] != binding[key]:
                raise RuntimeError(f"Window binding differs: {key}")
    else:
        save(binding_path, binding)
    records = [json.loads(line) for line in (root / "audit.jsonl").read_text().splitlines() if line.strip()]
    jobs, lineage, prepared = [], [], 0
    seen = set()
    for record in records:
        pool, eid = record["pool"], int(record["source_episode_index"])
        if pool not in pools or (episode_ids and eid not in episode_ids):
            continue
        if (pool, eid) in seen:
            raise RuntimeError("Duplicate parent audit record")
        seen.add((pool, eid))
        if record["status"] != "PREPARED_REVIEW_PENDING":
            continue
        if (
            record.get("source_revision") != "68651e4929d9fb00f798937b2d62617cab5c771d"
            or record.get("task") != "fold and stack the t-shirts"
            or record.get("fps") != FPS
        ):
            raise RuntimeError("Parent source revision/task/cadence differs from this window plan")
        if limit is not None and prepared >= limit:
            break
        n = int(record["length"])
        if set(record["videos"]) != set(CAMERAS):
            raise RuntimeError("Parent record must contain exactly three named cameras")
        for camera in CAMERAS:
            parent = record["videos"][camera]
            source = root / parent["artifact"]["path"]
            if digest(source) != parent["artifact"]["sha256"]:
                raise RuntimeError("Parent video identity changed")
        for index, (a, b, start, end) in enumerate(intervals(n)):
            window_id = f"tshirt-{pool}-{eid:06d}-w{index:03d}"
            directory = output / window_id
            directory.mkdir(exist_ok=True)
            receipt_path = directory / "receipt.json"
            expected = {
                "id": window_id,
                "pool": pool,
                "source_episode_index": eid,
                "parent_record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                "parent_from_frame": start,
                "parent_to_frame": end,
                "core_from_frame": a,
                "core_to_frame": b,
                "local_to_parent_offset_s": start / FPS,
                "fps": FPS,
                "source_revision": record["source_revision"],
                "training_episode": False,
                "review_status": "SEMANTIC_REVIEW_PENDING",
            }
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if any(receipt.get(k) != v for k, v in expected.items()):
                    raise RuntimeError("Existing window binding changed")
                if set(receipt["cameras"]) != set(CAMERAS):
                    raise RuntimeError("Existing window camera set changed")
                for camera, item in receipt["cameras"].items():
                    if (
                        item["path"] != str((directory / f"{camera}.mp4").relative_to(output))
                        or item["frames"] != end - start
                    ):
                        raise RuntimeError("Existing window camera binding changed")
                    if digest(output / item["path"]) != item["sha256"]:
                        raise RuntimeError("Existing window changed; original retained")
            else:
                receipt = dict(expected, cameras={})
                for camera in CAMERAS:
                    parent = record["videos"][camera]
                    source = root / parent["artifact"]["path"]
                    path = directory / f"{camera}.mp4"
                    camera_receipt = directory / f"{camera}.receipt.json"
                    camera_binding = dict(expected, camera=camera, parent_sha256=parent["artifact"]["sha256"])
                    if camera_receipt.exists():
                        item = json.loads(camera_receipt.read_text())
                        if item["binding"] != camera_binding or not path.exists() or digest(path) != item["sha256"]:
                            raise RuntimeError("Annotation camera receipt/file changed")
                    else:
                        if path.exists():
                            raise RuntimeError(
                                f"Unreceipted window exists: {path}; retain and use a new output version"
                            )
                        remux(source, path, Fraction(str(parent["offset_s"])) + Fraction(start, FPS), end - start)
                        item = {
                            "path": str(path.relative_to(output)),
                            "sha256": digest(path),
                            "frames": end - start,
                            "binding": camera_binding,
                        }
                        save(camera_receipt, item)
                    receipt["cameras"][camera] = {k: item[k] for k in ("path", "sha256", "frames")}
                save(receipt_path, receipt)
            job = {
                "id": window_id,
                "video": str(directory / "top.mp4"),
                "left_wrist": str(directory / "left_wrist.mp4"),
                "right_wrist": str(directory / "right_wrist.mp4"),
                "profile": "yam",
                "instruction": "fold and stack the t-shirts",
            }
            # Same CPU validation as native infer.py --validate-only, all three views.
            videos, profile = DataEnginePipeline._inputs(  # noqa: SLF001 -- use the native validation implementation
                **{k: v for k, v in job.items() if k != "id"}, yam32=False
            )
            if profile != "YAM" or len(videos) != 3:
                raise RuntimeError("Native FLORA profile validation failed")
            for video in videos.values():
                if (
                    video.rate != FPS
                    or len(video.pts_us) != end - start
                    or abs(video.duration_s - (end - start) / FPS) > 1e-5
                ):
                    raise RuntimeError("Native FLORA probe disagrees with planned frame interval")
            jobs.append(job)
            lineage.append(receipt)
            print(
                json.dumps({"event": "annotation_window_validated", "id": window_id, "frames": end - start}), flush=True
            )
        prepared += 1
    manifest = output / "jobs.jsonl"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    temporary.replace(manifest)
    save(
        output / "lineage.json",
        {
            "status": "NATIVE_INPUTS_VALIDATED_SEMANTIC_REVIEW_PENDING",
            "parent_episodes": prepared,
            "annotation_windows": len(jobs),
            "windows": lineage,
        },
    )
    print(
        json.dumps(
            {"event": "annotation_inputs_ready", "parents": prepared, "windows": len(jobs), "manifest": str(manifest)}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--flora-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pools", nargs="+", default=["train", "dev"])
    parser.add_argument("--episode-ids", nargs="*", type=int, default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / ".window-preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(
            args.root.resolve(),
            args.flora_root.resolve(),
            args.output.resolve(),
            args.pools,
            set(args.episode_ids),
            args.limit,
        )


if __name__ == "__main__":
    main()
