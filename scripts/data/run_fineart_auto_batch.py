#!/usr/bin/env python3
"""Chain existing FineART annotation CLIs for one frozen batch.

No training loop or semantic algorithm. Run with the existing Flora interpreter;
the caller supplies its model/GDN environment and host-level scheduling. Only
the selector acquires the shared GPU lock. Source datasets/results are read-only.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from datetime import UTC
from datetime import datetime
import hashlib
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

FPS = 30
SCHEMA = "condapi_fineart_auto_batch_v1"
TOOLS = (
    "prepare_fineart_subtask_review.py",
    "auto_fineart_timeline.py",
    "select_fineart_candidates.py",
    "materialize_fineart_language.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Replace only the caller-owned run status; never source inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fchmod(stream.fileno(), 0o644)
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_stable(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    require(
        (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"Input changed while freezing: {path}; original left untouched",
    )
    return content


def exclusive_bytes(path, content):
    """Publish new bytes or verify an identical content-addressed snapshot."""
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == content, f"Existing snapshot has different bytes: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fchmod(stream.fileno(), 0o644)
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == content, f"Concurrent snapshot differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def freeze_sibling(path):
    path = Path(path).resolve(strict=True)
    content = read_stable(path)
    sha = hashlib.sha256(content).hexdigest()
    target = path.with_name(f"{path.stem}.fineart-snapshot-{sha[:20]}{path.suffix}")
    exclusive_bytes(target, content)
    require(digest(target) == sha, "Frozen sibling hash mismatch")
    return {"original_path": str(path), "path": str(target), "sha256": sha}


def freeze_tools(directory, output, *, skip_materialize=False):
    """Keep native auto->../review import layout, including flat deployments."""
    directory = Path(directory).resolve(strict=True)
    paths, bindings = {}, {}
    for name in TOOLS:
        original = directory / name
        if name == "prepare_fineart_subtask_review.py" and not original.is_file():
            original = directory.parent / name
        target = output / "code" / (name if name == "prepare_fineart_subtask_review.py" else "data/" + name)
        if name == "materialize_fineart_language.py" and skip_materialize and not original.is_file():
            paths[name] = target
            bindings[name] = {"status": "SKIPPED_NOT_DEPLOYED", "reason": "--skip-materialize"}
            continue
        require(original.is_file(), f"Missing CLI {name} under --tools-dir")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = read_stable(original)
        exclusive_bytes(target, content)
        paths[name] = target
        bindings[name] = {
            "original_path": str(original.resolve()),
            "path": str(target.resolve()),
            "sha256": digest(target),
        }
    return paths, bindings


def extract_actual_timestamps(draft_file, audit_file, output):
    """CPU-only real Parquet column extraction; never substitute frame/FPS."""
    import pyarrow.parquet as pq  # noqa: PLC0415 - optional CPU dependency, absent from protocol-only runs

    draft, audit_file = load(draft_file), Path(audit_file)
    metadata = {}
    for line in audit_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["pool"], row["source_episode_index"])
        require(key not in metadata, f"Duplicate parent audit {key}")
        metadata[key] = row
    parents = []
    for parent in draft["parents"]:
        key = (parent["pool"], parent["source_episode_index"])
        meta = metadata[key]
        length = parent["length"]
        require(
            meta["fps"] == parent["fps"] == FPS and meta["length"] == length and length > 0,
            f"Parent frame/FPS metadata mismatch: {key}",
        )
        record_sha = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        require(
            all(w["parent_record_sha256"] == record_sha for w in parent["windows"]),
            "Draft is bound to another parent audit record",
        )
        path = Path(meta["data"]["path"])
        path = path.resolve() if path.is_absolute() else (audit_file.parent / path).resolve()
        expected = meta["data"]["sha256"]
        require(digest(path) == expected, f"Original parent Parquet hash mismatch: {path}")
        table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp"])
        require(table.num_rows == length, f"Original Parquet frame count differs: {key}")
        require(table["episode_index"].to_pylist() == [key[1]] * length, f"Wrong source episode rows: {key}")
        require(table["frame_index"].to_pylist() == list(range(length)), f"Nonconsecutive parent frame_index: {key}")
        timestamps = table["timestamp"].to_pylist()
        require(
            all(type(t) in (float, int) and math.isfinite(t) for t in timestamps), "Invalid actual source timestamp"
        )
        require(all(b > a for a, b in pairwise(timestamps)), "Actual source timestamps are not increasing")
        require(
            all(abs(t - (timestamps[0] + i / FPS)) <= 1e-4 for i, t in enumerate(timestamps)),
            "Actual source timestamp gap / FPS mismatch",
        )
        require(digest(path) == expected, "Original Parquet changed during timestamp extraction")
        parents.append(
            {
                "pool": key[0],
                "source_episode_index": key[1],
                "fps": FPS,
                "timestamps": timestamps,
                "timestamp_source": {
                    "kind": "actual_parquet_timestamp_column",
                    "column": "timestamp",
                    "path": str(path),
                    "sha256": expected,
                    "extraction_method": "pyarrow.read_table; actual episode_index/frame_index/timestamp columns; full parent identity/order checked",
                },
            }
        )
    require(parents, "No parent timestamps to export")
    value = {
        "schema_version": 1,
        "parents": parents,
        "training_ready": False,
        "draft_source": {"path": str(Path(draft_file).resolve()), "sha256": digest(draft_file)},
        "parent_audit_source": {"path": str(audit_file.resolve()), "sha256": digest(audit_file)},
    }
    write_json(output, value)
    return {"parents": len(parents), "frames": sum(len(p["timestamps"]) for p in parents), "output": str(output)}


class Batch:
    def __init__(self, output, runner):
        self.output, self.runner, self.stages = output, runner, []
        self.current = "initialization"

    def status(self, state, **values):
        write_json(
            self.output / "status.json",
            {
                "schema_version": SCHEMA,
                "status": state,
                "updated_at": now(),
                "current_stage": self.current,
                "stages": self.stages,
                "training_ready": False,
                "dataset_release": False,
                **values,
            },
        )

    def stage(self, name, *, argv=None, function=None, skip=None):
        self.current = name
        index = len(self.stages)
        base = f"{index:02d}-{name}"
        stdout_path, stderr_path = (
            self.output / "logs" / (base + ".stdout.log"),
            self.output / "logs" / (base + ".stderr.log"),
        )
        stdout_path.parent.mkdir(exist_ok=True)
        record = {
            "name": name,
            "started_at": now(),
            "argv": [str(x) for x in argv] if argv else None,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "status": "RUNNING",
        }
        self.stages.append(record)
        self.status("RUNNING")
        started = time.monotonic()
        try:
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                if skip:
                    record.update(status="SKIPPED", reason=skip, exit_code=0)
                    stdout.write(json.dumps({"status": "SKIPPED", "reason": skip}) + "\n")
                elif function is not None:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        value = function()
                        if value is not None:
                            print(json.dumps(value, ensure_ascii=False))
                    record.update(status="COMPLETED", exit_code=0)
                else:
                    result = self.runner([str(x) for x in argv], stdout=stdout, stderr=stderr, check=False)
                    record.update(
                        status="COMPLETED" if result.returncode == 0 else "FAILED", exit_code=result.returncode
                    )
                    require(result.returncode == 0, f"Stage {name} exited {result.returncode}; see {stderr_path}")
        except Exception as error:
            record.update(status="FAILED", error=str(error), exit_code=record.get("exit_code", 2))
            with stderr_path.open("a") as stderr:
                traceback.print_exc(file=stderr)
            raise
        finally:
            record.update(finished_at=now(), elapsed_s=time.monotonic() - started)
            write_json(self.output / "stages" / (base + ".json"), record)
            self.status("FAILED" if record["status"] == "FAILED" else "RUNNING")


def run(args, *, runner=subprocess.run, timestamp_extractor=extract_actual_timestamps):
    output = args.output_root.expanduser().resolve()
    require(not output.exists(), "--output-root must be NEW; failed/completed artifacts are retained")
    require(
        args.unknown_mode == "native-empty-anchor-fallback" or args.skip_materialize,
        "strict-parent currently requires --skip-materialize; materializer consumes fallback views only",
    )
    output.mkdir(parents=True)
    batch = Batch(output, runner)
    batch.status("INITIALIZING")
    try:
        state = {}

        def freeze():
            lineage, audit = freeze_sibling(args.lineage), freeze_sibling(args.parent_audit)
            tools, tool_bindings = freeze_tools(args.tools_dir, output, skip_materialize=args.skip_materialize)
            policy = output / "inputs/policy.json"
            policy.parent.mkdir(parents=True, exist_ok=True)
            exclusive_bytes(policy, read_stable(args.policy))
            require(args.raw_results.resolve().is_dir(), "--raw-results directory does not exist")
            state.update(lineage=Path(lineage["path"]), audit=Path(audit["path"]), policy=policy, tools=tools)
            ownership = {
                "schema_version": SCHEMA,
                "created_at": now(),
                "lineage": lineage,
                "parent_audit": audit,
                "policy": {"original_path": str(args.policy.resolve()), "path": str(policy), "sha256": digest(policy)},
                "tools": tool_bindings,
                "orchestrator_sha256": digest(__file__),
                "python": sys.executable,
                "raw_results": str(args.raw_results.resolve()),
                "model_dir": str(args.model_dir.resolve()),
                "flora_root": str(args.flora_root.resolve()),
                "unknown_mode": args.unknown_mode,
                "lock_file_passed_to_selector": str(args.lock_file),
                "environment_inherited_not_logged": True,
                "training_ready": False,
                "dataset_release": False,
            }
            write_json(output / "ownership.json", ownership)
            return {"snapshots_frozen": True, "source_files_modified": False}

        batch.stage("freeze-inputs-and-tools", function=freeze)
        py, tools = sys.executable, state["tools"]
        draft, plan, timestamps, view = [
            output / x for x in ("draft.json", "plan.json", "actual-timestamps.json", "auto-view.json")
        ]
        batch.stage(
            "draft",
            argv=[
                py,
                tools[TOOLS[0]],
                "--draft",
                "--lineage",
                state["lineage"],
                "--parent-metadata",
                state["audit"],
                "--results-dir",
                args.raw_results.resolve(),
                "--output",
                draft,
            ],
        )
        batch.stage(
            "plan",
            argv=[py, tools[TOOLS[1]], "--plan", "--draft-file", draft, "--policy", state["policy"], "--output", plan],
        )
        tasks = load(plan)["tasks"]
        require(isinstance(tasks, list), "Invalid plan tasks envelope")
        batch.stage("actual-timestamps", function=lambda: timestamp_extractor(draft, state["audit"], timestamps))
        decisions = output / "selector/decisions.json"
        batch.stage(
            "selector",
            argv=[
                py,
                tools[TOOLS[2]],
                "--tasks",
                plan,
                "--model-dir",
                args.model_dir.resolve(),
                "--flora-root",
                args.flora_root.resolve(),
                "--output-dir",
                decisions.parent,
                "--lock-file",
                args.lock_file,
            ],
            skip=None if tasks else "No selector tasks; eligible rules resolved this batch",
        )
        apply_argv = [
            py,
            tools[TOOLS[1]],
            "--apply",
            "--plan-file",
            plan,
            "--policy",
            state["policy"],
            "--frame-timestamps",
            timestamps,
            "--unknown-mode",
            args.unknown_mode,
            "--output",
            view,
        ]
        if tasks:
            require(decisions.is_file(), "Selector exited without decisions.json")
            apply_argv += ["--decisions", decisions]
        batch.stage("apply", argv=apply_argv)
        applied = load(view)
        materialized = output / "materialized"
        reason = (
            "--skip-materialize requested"
            if args.skip_materialize
            else ("No eligible parents; retain explicit local-repair queue" if not applied.get("parents") else None)
        )
        batch.stage(
            "materialize", argv=[py, tools[TOOLS[3]], "--view", view, "--output-root", materialized], skip=reason
        )
        manifest = materialized / "manifest.json"
        result = {
            "schema_version": SCHEMA,
            "status": "LANGUAGE_COLUMNS_MATERIALIZED_NOT_DATASET_RELEASED"
            if manifest.is_file()
            else "ANNOTATION_VIEW_NOT_MATERIALIZED_NOT_TRAINING_READY",
            "training_ready": False,
            "dataset_release": False,
            "task_success": "NOT_REVIEWED",
            "output_root": str(output),
            "selector_tasks": len(tasks),
            "unknown_mode": args.unknown_mode,
            "artifacts": {
                name: {"path": str(path), "sha256": digest(path)}
                for name, path in (
                    ("draft", draft),
                    ("plan", plan),
                    ("actual_timestamps", timestamps),
                    ("auto_view", view),
                    ("decisions", decisions),
                    ("materialization_manifest", manifest),
                )
                if path.is_file()
            },
            "view_status": applied.get("status"),
            "coverage": applied.get("coverage"),
            "pending_parent_count": applied.get("pending_queue", {}).get(
                "parents", len(applied.get("pending_parents", []))
            ),
            "stages": batch.stages,
        }
        if manifest.is_file():
            value = load(manifest)
            require(
                value.get("status") == "LANGUAGE_COLUMNS_MATERIALIZED_NOT_DATASET_RELEASED"
                and value.get("training_ready") is False
                and value.get("dataset_release") is False,
                "Materializer must not claim complete dataset/training readiness",
            )
            result["materialization_counts"] = value.get("counts")
        write_json(output / "summary.json", result)
        batch.status(result["status"], summary=str(output / "summary.json"))
        return result
    except Exception as error:
        failure = {
            "schema_version": SCHEMA,
            "status": "BATCH_FAILED_ARTIFACTS_RETAINED",
            "stage": batch.current,
            "error": str(error),
            "failed_at": now(),
            "training_ready": False,
            "dataset_release": False,
        }
        write_json(output / "failure.json", failure)
        batch.status(failure["status"], error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lineage", "parent-audit", "raw-results", "policy", "model-dir", "flora-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--lock-file", type=Path, default=Path("/data/.flora-annotation-gpu.lock"))
    parser.add_argument(
        "--unknown-mode",
        choices=("strict-parent", "native-empty-anchor-fallback"),
        default="native-empty-anchor-fallback",
    )
    parser.add_argument("--skip-materialize", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(json.dumps({"status": result["status"], "output_root": result["output_root"], "training_ready": False}))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error), "training_ready": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
