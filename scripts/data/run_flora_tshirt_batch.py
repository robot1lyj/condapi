#!/usr/bin/env python3
"""Run unchanged native FLORA stages with validated, exclusive checkpoints.

This is orchestration only: models, prompts, frame plans, processors, precision,
and generation remain in the upstream Flora checkout. No training is performed.
"""

from __future__ import annotations

# Native ML imports stay lazy so --validate-only has no model dependency.
# ruff: noqa: PLC0415
import argparse
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
import fcntl
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback

CHECKPOINT_SCHEMA = "condapi_flora_stage_v1"
NATIVE_SCHEMA = "flora_demo_annotation_v1"


def event(name, **values):
    print(
        "FLORA_BATCH "
        + json.dumps({"event": name, "utc": datetime.now(UTC).isoformat(), **values}, sort_keys=True, allow_nan=False),
        flush=True,
    )


def progress_event(fraction, desc="", *, job_id, stage):
    event("progress", id=job_id, stage=stage, fraction=fraction, detail=desc)


def payload_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_exclusive(path, value):
    """Publish a fully fsynced file atomically; never replace another result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="." + path.name + ".", suffix=".partial", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload_bytes(value))
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def worker_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another FLORA batch worker holds {path}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "utc": datetime.now(UTC).isoformat()}) + "\n")
        stream.flush()
        event("worker_lock_acquired", path=str(path), pid=os.getpid())
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_jobs(path, limit):
    source = path.expanduser().resolve(strict=True)
    allowed = {"id", "video", "profile", "left_wrist", "right_wrist", "yam32", "instruction"}
    jobs, names, artifact_names = [], set(), set()
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - allowed or "video" not in row:
            raise ValueError(f"Invalid native JSONL row {line_number}; put lineage in a separate sidecar.")
        name = row.pop("id", f"video_{line_number:06d}")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name) or name in names:
            raise ValueError(f"Invalid or duplicate job id on row {line_number}.")
        names.add(name)
        artifacts = {name + suffix for suffix in (".json", ".tempseg.json", ".captioner.json")}
        if artifacts & artifact_names:
            raise ValueError(f"Job id collides with another job's stage artifact on row {line_number}.")
        artifact_names.update(artifacts)
        kwargs = {"profile": "yam", "left_wrist": None, "right_wrist": None, "yam32": False, "instruction": ""}
        kwargs.update(row)
        if kwargs["profile"] != "yam" or not kwargs["left_wrist"] or not kwargs["right_wrist"]:
            raise ValueError("This T-shirt runner requires native profile=yam and all three camera files.")
        for key in ("video", "left_wrist", "right_wrist"):
            if not isinstance(kwargs[key], str) or not kwargs[key].strip():
                raise ValueError(f"Invalid {key} path on row {line_number}.")
            value = Path(kwargs[key]).expanduser()
            kwargs[key] = (value if value.is_absolute() else source.parent / value).resolve(strict=True)
        jobs.append({"id": name, "kwargs": kwargs})
    if not jobs:
        raise ValueError("No jobs in the JSONL.")
    return jobs[:limit] if limit is not None else jobs


def stamps(paths):
    values = {}
    for path in paths:
        stat = path.stat()
        values[str(path)] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return values


def assert_sources_unchanged(job):
    if stamps(job["source_paths"]) != job["source_stamps"]:
        raise RuntimeError(f"Source media changed during this run: {job['id']}; stop and revalidate.")


def prepare_jobs(jobs, engine, models):
    """Native CPU decode/probe, input SHA and synchronization checks, once/run."""
    for job in jobs:
        started = time.monotonic()
        paths = [job["kwargs"][key] for key in ("video", "left_wrist", "right_wrist")]
        before = stamps(paths)
        videos, profile = engine._inputs(**job["kwargs"])  # noqa: SLF001 -- use the native validation implementation
        after = stamps(paths)
        if before != after:
            raise RuntimeError(f"Source media changed during validation: {job['id']}")
        job.update(videos=videos, profile=profile, source_paths=paths, source_stamps=after)
        job["identity"] = {
            "input_sha256": {role: video.sha256 for role, video in videos.items()},
            "model_revisions": models,
            "profile": profile,
            "yam32": job["kwargs"]["yam32"],
            "instruction": job["kwargs"]["instruction"],
            "camera_roles": list(videos),
            "duration_s": videos["main"].duration_s,
        }
        job["validation_s"] = time.monotonic() - started
        event(
            "input_validated",
            id=job["id"],
            duration_s=videos["main"].duration_s,
            validation_wall_s=job["validation_s"],
            input_sha256=job["identity"]["input_sha256"],
            new_model_calls=0,
        )


def read_checkpoint(path, job, stage):
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA
        or value.get("stage") != stage
        or value.get("id") != job["id"]
        or value.get("identity") != job["identity"]
    ):
        raise RuntimeError(f"Checkpoint contract/hash/revision mismatch: {path}")
    return value


def validate_intervals(rows, duration):
    for row in rows:
        start, end = row["start_s"], row["end_s"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.001)
        ):
            raise RuntimeError("Native FLORA returned an invalid interval.")


def validate_caption_checkpoint(checkpoint, source, source_path, job):
    if checkpoint.get("tempseg_sha256") != file_sha256(source_path):
        raise RuntimeError(f"Caption checkpoint does not match TempSeg artifact: {source_path}")
    result = checkpoint["native_result"]
    identity = job["identity"]
    if result.get("schema_version") != NATIVE_SCHEMA or any(
        result.get(key) != identity[key]
        for key in ("duration_s", "profile", "camera_roles", "input_sha256", "model_revisions")
    ):
        raise RuntimeError("Caption checkpoint has inconsistent native output identity.")
    if result["pipeline"]["temporal_windows"] != source["temporal_audit"]:
        raise RuntimeError("Caption checkpoint has different temporal provenance.")
    validate_intervals(result["segments"], identity["duration_s"])
    if len(result["segments"]) != len(source["segments"]):
        raise RuntimeError("Caption checkpoint changed the native segment count.")
    for row, segment in zip(result["segments"], source["segments"], strict=True):
        if row["start_s"] != segment["start_s"] or row["end_s"] != segment["end_s"]:
            raise RuntimeError("Caption checkpoint changed native segment boundaries.")
    return result


def publish_or_verify_final(path, result):
    expected = payload_bytes(result)
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"Existing final result differs from validated caption checkpoint: {path}")
        event("final_reused", output=str(path))
        return
    publish_exclusive(path, result)
    event("final_published", output=str(path), sha256=hashlib.sha256(expected).hexdigest())


def session_class(native_session, torch, mode):
    class TimedSession(native_session):
        def __init__(self):
            super().__init__()
            self.loaded_kind = None
            self.total_load_wall_s = 0.0
            self.load_calls = self.reused_calls = 0

        def load(self, kind):
            if (
                mode == "session-cache"
                and self.loaded_kind == kind
                and self.model is not None
                and self.proc is not None
            ):
                self.reused_calls += 1
                event("model_reused", kind=kind, mode=mode)
                return self.model, self.proc
            torch.cuda.synchronize()
            started = time.monotonic()
            result = super().load(kind)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            self.total_load_wall_s += elapsed
            self.load_calls += 1
            self.loaded_kind = kind
            event("model_load_measured", kind=kind, mode=mode, load_wall_s=elapsed)
            return result

        def close(self):
            self.loaded_kind = None
            super().close()

    return TimedSession


def stage_timing(session, torch, operation):
    """Synchronized wall time; inference remainder includes native CPU work."""
    torch.cuda.synchronize()
    load_before = session.total_load_wall_s
    calls_before, reused_before = session.load_calls, session.reused_calls
    started = time.monotonic()
    result = operation()
    torch.cuda.synchronize()
    wall = time.monotonic() - started
    load = session.total_load_wall_s - load_before
    return result, {
        "stage_wall_s": wall,
        "model_load_wall_s": load,
        "native_work_wall_s": wall - load,
        "actual_model_loads": session.load_calls - calls_before,
        "reused_model_loads": session.reused_calls - reused_before,
    }


def run_stages(args, jobs, native_session, segment, caption_three, torch):
    timed_session = session_class(native_session, torch, args.mode)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A final file is only resumable when its identity-bound caption checkpoint
    # is present. This deliberately rejects unverified legacy/foreign outputs.
    for job in jobs:
        final = output / (job["id"] + ".json")
        if final.exists() and not (output / (job["id"] + ".captioner.json")).exists():
            raise RuntimeError(f"Final result has no identity-bound caption checkpoint: {final}")

    if args.stage in ("all", "tempseg"):
        session = timed_session()
        try:
            for job in jobs:
                assert_sources_unchanged(job)
                path = output / (job["id"] + ".tempseg.json")
                if path.exists():
                    source = read_checkpoint(path, job, "tempseg")
                    validate_intervals(source["segments"], job["identity"]["duration_s"])
                    event("tempseg_reused", id=job["id"], checkpoint=str(path))
                    continue
                pool = "yam32" if job["kwargs"]["yam32"] else "yam"
                progress = partial(progress_event, job_id=job["id"], stage="tempseg")
                operation = partial(segment, job["videos"], pool, session, job["kwargs"]["instruction"], progress)
                (segments, audit), timing = stage_timing(session, torch, operation)
                assert_sources_unchanged(job)
                validate_intervals(segments, job["identity"]["duration_s"])
                publish_exclusive(
                    path,
                    {
                        "schema_version": CHECKPOINT_SCHEMA,
                        "stage": "tempseg",
                        "id": job["id"],
                        "identity": job["identity"],
                        "segments": segments,
                        "temporal_audit": audit,
                        "timing": timing,
                        "mode": args.mode,
                    },
                )
                event("tempseg_complete", id=job["id"], segments=len(segments), checkpoint=str(path), **timing)
        finally:
            session.close()

    if args.stage in ("all", "captioner"):
        # Fail before loading Captioner if any required segmentation is absent
        # or belongs to different inputs/instructions/model revisions.
        for job in jobs:
            path = output / (job["id"] + ".tempseg.json")
            source = read_checkpoint(path, job, "tempseg")
            validate_intervals(source["segments"], job["identity"]["duration_s"])
        session = timed_session()
        try:
            for job in jobs:
                assert_sources_unchanged(job)
                source_path = output / (job["id"] + ".tempseg.json")
                source = read_checkpoint(source_path, job, "tempseg")
                path = output / (job["id"] + ".captioner.json")
                final = output / (job["id"] + ".json")
                if path.exists():
                    checkpoint = read_checkpoint(path, job, "captioner")
                    result = validate_caption_checkpoint(checkpoint, source, source_path, job)
                    publish_or_verify_final(final, result)
                    event("captioner_reused", id=job["id"], checkpoint=str(path))
                    continue
                progress = partial(progress_event, job_id=job["id"], stage="captioner")
                operation = partial(
                    caption_three,
                    job["videos"],
                    source["segments"],
                    session,
                    job["kwargs"]["instruction"],
                    progress,
                    profile=job["profile"],
                )
                (captions, audit), timing = stage_timing(session, torch, operation)
                assert_sources_unchanged(job)
                for row, original in zip(captions, source["segments"], strict=True):
                    row["segmentation_provenance"] = original["metadata"].get("segmentation_provenance", "fine")
                validate_intervals(captions, job["identity"]["duration_s"])
                identity = job["identity"]
                result = {
                    "schema_version": NATIVE_SCHEMA,
                    "duration_s": identity["duration_s"],
                    "segments": captions,
                    "elapsed_s": round(source["timing"]["stage_wall_s"] + timing["stage_wall_s"], 2),
                    "source_clock": "seconds relative to the first native source presentation timestamp",
                    "camera_roles": identity["camera_roles"],
                    "profile": identity["profile"],
                    "model_revisions": identity["model_revisions"],
                    "input_sha256": identity["input_sha256"],
                    "pipeline": {"temporal_windows": source["temporal_audit"], "caption_inputs": audit},
                }
                checkpoint = {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "stage": "captioner",
                    "id": job["id"],
                    "identity": identity,
                    "tempseg_sha256": file_sha256(source_path),
                    "native_result": result,
                    "timing": timing,
                    "mode": args.mode,
                }
                validate_caption_checkpoint(checkpoint, source, source_path, job)
                publish_exclusive(path, checkpoint)
                publish_or_verify_final(final, result)
                event(
                    "captioner_complete",
                    id=job["id"],
                    segments=len(captions),
                    incomplete=sum(row.get("status") != "complete" for row in captions),
                    checkpoint=str(path),
                    **timing,
                )
        finally:
            session.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--flora-root",
        type=Path,
        required=True,
        help="Unmodified upstream Flora checkout, containing captioner/batch_utils.",
    )
    result.add_argument("--input-jsonl", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--mode", choices=("baseline", "session-cache"), default="session-cache")
    result.add_argument("--stage", choices=("all", "tempseg", "captioner"), default="all")
    result.add_argument("--limit", type=int, help="Only the first N manifest jobs, in manifest order.")
    result.add_argument(
        "--lock-file",
        type=Path,
        default=Path("/data/.flora-annotation-gpu.lock"),
        help="All FLORA workers must share this file on a shared mount.",
    )
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="Native CPU media validation only; no model import/load, lock, output files or GPU calls.",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    root = args.flora_root.expanduser().resolve(strict=True)
    if not (root / "captioner" / "batch_utils" / "api.py").is_file():
        raise ValueError("--flora-root must contain captioner/batch_utils/api.py.")
    jobs = load_jobs(args.input_jsonl, args.limit)
    sys.path.insert(0, str(root / "captioner"))
    from batch_utils.api import DataEnginePipeline
    from batch_utils.config import MODELS

    for spec in MODELS.values():
        variable = spec.get("repo_env")
        if variable and os.environ.get(variable, spec["repo"]) != spec["repo"]:
            raise ValueError(f"{variable} must preserve the official pinned model repository.")
    engine = DataEnginePipeline()
    if args.validate_only:
        prepare_jobs(jobs, engine, MODELS)
        event("validation_complete", jobs=len(jobs), new_model_calls=0)
        return 0
    with worker_lock(args.lock_file.expanduser().resolve()):
        prepare_jobs(jobs, engine, MODELS)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for annotation; use --validate-only for CPU media validation.")
        from batch_utils.models import Session
        from batch_utils.pipeline import caption_three
        from batch_utils.pipeline import segment

        event(
            "run_start",
            jobs=len(jobs),
            stage=args.stage,
            mode=args.mode,
            torch_version=torch.__version__,
            gpu=torch.cuda.get_device_name(0),
            flora_root=str(root),
            manifest_sha256=file_sha256(args.input_jsonl.expanduser().resolve()),
        )
        started = time.monotonic()
        run_stages(args, jobs, Session, segment, caption_three, torch)
        event("run_complete", jobs=len(jobs), stage=args.stage, mode=args.mode, wall_s=time.monotonic() - started)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        event("run_failed", error=str(error), error_type=type(error).__name__)
        traceback.print_exc()
        raise SystemExit(1) from error
