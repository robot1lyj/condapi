#!/usr/bin/env python3
"""Wrap an explicitly authorized Slurm command with per-attempt diagnostics.

Uses the original trainer/environment/checkpoint contract. Never retries a failed
trainer. Only its diagnostic child processes are stopped during cleanup.
"""

import argparse
import contextlib
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from collect_training_incident import collect
from collect_training_incident import selected_environment


def now():
    return datetime.now(UTC).isoformat()


def diagnostic_environment(output, base):
    env = dict(base)
    env.update(
        PYTHONFAULTHANDLER="1",
        PYTHONUNBUFFERED="1",
        NCCL_DEBUG="INFO",
        NCCL_DEBUG_SUBSYS="INIT,ENV,GRAPH,COLL,P2P,SHM,NET",
        NCCL_DEBUG_FILE=str(output / "nccl.%h.%p.log"),
        CUDA_ENABLE_COREDUMP_ON_EXCEPTION="1",
        CUDA_COREDUMP_FILE=str(output / "cuda.%h.%p.%t.nvcudmp"),
    )
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_dump_to=" + str(output / "xla") + " --xla_dump_hlo_as_text"
    ).strip()
    return env


def process_snapshot(pid):
    proc = Path("/proc") / str(pid)
    result = {"utc": now(), "pid": pid}
    try:
        result["stat"] = (proc / "stat").read_text()
        result["environment"] = selected_environment((proc / "environ").read_bytes())
        result["libraries"] = sorted(
            {
                line.split()[-1]
                for line in (proc / "maps").read_text().splitlines()
                if any(key in line for key in ("libcuda", "libnccl", "libcudnn", "libcublas", "jaxlib", "cuda_plugin"))
            }
        )
        result["limits"] = (proc / "limits").read_text()
        result["cgroup"] = (proc / "cgroup").read_text()
    except OSError as exc:
        result["error"] = str(exc)
    return result


def host_sample(pid):
    result = {"utc": now(), "trainer_pid": pid}
    for key, path in {"meminfo": "/proc/meminfo", "loadavg": "/proc/loadavg"}.items():
        try:
            result[key] = Path(path).read_text()
        except OSError as exc:
            result[key] = {"error": str(exc)}
    result["edac"] = {}
    for pattern in ("mc*/ce_count", "mc*/ue_count"):
        for path in Path("/sys/devices/system/edac/mc").glob(pattern):
            try:
                result["edac"][str(path)] = path.read_text().strip()
            except OSError as exc:
                result["edac"][str(path)] = {"error": str(exc)}
    return result


def run(args):
    output = args.output.resolve()
    if any(char.isspace() for char in str(output)):
        raise ValueError("XLA dump directory must not contain whitespace")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "xla").mkdir()
    env = diagnostic_environment(output, os.environ)
    event_file = (output / "events.jsonl").open("x", buffering=1)

    def event(kind, **fields):
        event_file.write(json.dumps({"utc": now(), "event": kind, **fields}) + "\n")

    snapshot_args = SimpleNamespace(
        output=output / "before",
        since=args.since,
        job_id=os.environ.get("SLURM_JOB_ID"),
        pid=None,
        python=args.metadata_python,
        log=args.source_file,
        timeout=10,
    )
    event("starting", command=args.command, cwd=str(Path.cwd()), job_id=snapshot_args.job_id)
    collect(snapshot_args)
    monitors = []
    handles = []
    trainer = None
    previous_handlers = {}
    received_signals = []

    def forward_signal(signum, _frame):
        received_signals.append(signum)
        event("wrapper_signal", signal=signum)
        if trainer is not None and trainer.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(trainer.pid, signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous_handlers[sig] = signal.signal(sig, forward_signal)
    try:
        for name, command in {
            "gpu-telemetry": [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,pci.bus_id,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
                "--format=csv",
                "--loop=5",
            ],
            "kernel-follow": ["dmesg", "--follow", "--time-format", "iso"],
        }.items():
            out = (output / (name + ".log")).open("xb")
            err = (output / (name + ".stderr.txt")).open("xb")
            handles.extend((out, err))
            try:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True
                )
                monitors.append((name, process))
                event("monitor_started", name=name, pid=process.pid)
            except OSError as exc:
                event("monitor_unavailable", name=name, error=str(exc))
        if received_signals:
            event("cancelled_before_launch", signals=received_signals)
            return 128 + received_signals[-1]
        with (
            (output / "trainer.log").open("xb") as train_log,
            (output / "host-telemetry.jsonl").open("x", buffering=1) as host_log,
        ):
            trainer = subprocess.Popen(
                args.command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=train_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            event("trainer_started", pid=trainer.pid)
            next_process_snapshot = 0.0
            while True:
                host_log.write(json.dumps(host_sample(trainer.pid)) + "\n")
                if time.monotonic() >= next_process_snapshot:
                    with (output / "process-snapshots.jsonl").open("a") as handle:
                        handle.write(json.dumps(process_snapshot(trainer.pid)) + "\n")
                    next_process_snapshot = time.monotonic() + 60
                for name, process in monitors:
                    if process.poll() is not None and not getattr(process, "reported", False):
                        event("monitor_exited", name=name, returncode=process.returncode)
                        process.reported = True
                try:
                    code = trainer.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            event(
                "trainer_exited",
                pid=trainer.pid,
                returncode=code,
                signal=-code if code < 0 else None,
                received_signals=received_signals,
            )
            (output / "exit.json").write_text(
                json.dumps(
                    {
                        "utc": now(),
                        "pid": trainer.pid,
                        "returncode": code,
                        "signal": -code if code < 0 else None,
                        "wrapper_received_signals": received_signals,
                    },
                    indent=2,
                )
                + "\n"
            )
            snapshot_args.output = output / "after"
            snapshot_args.pid = trainer.pid
            # Keep originals in place; the collector hashes its independent copy.
            snapshot_args.log = [output / "trainer.log", *args.source_file]
            try:
                collect(snapshot_args)
                event("after_snapshot_complete")
            except Exception as exc:
                event("after_snapshot_failed", error=repr(exc))
            return code if code >= 0 else 128 - code
    finally:
        # If capture itself fails, do not leave an untracked trainer alive.
        if trainer is not None and trainer.poll() is None:
            os.killpg(trainer.pid, signal.SIGTERM)
            try:
                trainer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(trainer.pid, signal.SIGKILL)
                trainer.wait()
        for name, process in monitors:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            event("monitor_stopped", name=name, returncode=process.returncode)
        for handle in handles:
            handle.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        event_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--metadata-python", default=sys.executable)
    parser.add_argument("--source-file", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not os.environ.get("SLURM_JOB_ID") or not args.command:
        parser.error("An allocated Slurm job and an explicit authorized command are required")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
