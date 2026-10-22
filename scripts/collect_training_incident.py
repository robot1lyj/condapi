#!/usr/bin/env python3
"""Capture a read-only Linux training incident snapshot, using only the stdlib.

Run on the affected compute node. Does not launch training, signal processes,
change GPU settings, use sudo, or configure future jobs. See docs/03 for usage.
"""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess

# Never dump the whole environment: it can contain credentials.
ENV_KEYS = {
    "SLURM_JOB_ID",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_PROCID",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "LD_LIBRARY_PATH",
    "PYTHONFAULTHANDLER",
    "PYTHONUNBUFFERED",
    "XLA_FLAGS",
    "JAX_PLATFORMS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "NCCL_DEBUG",
    "NCCL_DEBUG_FILE",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_DEBUG_TIMESTAMP_FORMAT",
    "NCCL_DEBUG_TIMESTAMP_LEVELS",
    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
    "CUDA_COREDUMP_FILE",
    "CUDA_COREDUMP_GENERATION_FLAGS",
    "CUDA_LAUNCH_BLOCKING",
    "OPENPI_NUMERIC_DIAGNOSTICS_DIR",
    "OPENPI_JAX_CACHE_DIR",
}
METADATA_CODE = """
import importlib.metadata as m, json, sys
prefixes = ('jax', 'nvidia', 'torch', 'flax', 'orbax', 'numpy', 'triton', 'lerobot', 'av')
print(json.dumps({'python': sys.version, 'packages': {
    d.metadata['Name']: d.version for d in m.distributions()
    if d.metadata['Name'].lower().startswith(prefixes)
}}, indent=2))
"""


def utc_now():
    return datetime.now(UTC).isoformat()


def selected_environment(data):
    pairs = (part.split("=", 1) for part in data.decode(errors="replace").split("\0") if "=" in part)
    return {key: value for key, value in pairs if key in ENV_KEYS}


class Capture:
    def __init__(self, output, timeout=15):
        self.output = Path(output)
        self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.timeout = timeout
        self.records = []

    def command(self, name, argv):
        record = {"name": name, "argv": argv, "started_utc": utc_now()}
        with (
            (self.output / (name + ".stdout.txt")).open("xb") as out,
            (self.output / (name + ".stderr.txt")).open("xb") as err,
        ):
            try:
                result = subprocess.run(
                    argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=self.timeout, check=False
                )
                record.update(status="ok" if result.returncode == 0 else "command_failed", returncode=result.returncode)
            except subprocess.TimeoutExpired:
                record.update(status="timeout", returncode=None)
            except OSError as exc:
                record.update(status="unavailable", error=str(exc), returncode=None)
        record["finished_utc"] = utc_now()
        self.records.append(record)

    def snapshot(self, name, source):
        """Copy the initial extent; do not chase a growing log or block on a FIFO."""
        source = Path(source)
        record = {"name": name, "source": str(source), "started_utc": utc_now()}
        try:
            # O_NONBLOCK also protects against accidentally supplied special files.
            fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as src:
                before = os.fstat(src.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("Only regular files can be snapshotted")
                digest = hashlib.sha256()
                count = 0
                with (self.output / name).open("xb") as dst:
                    remaining = before.st_size
                    while remaining:
                        data = src.read(min(1024 * 1024, remaining))
                        if not data:
                            break
                        dst.write(data)
                        digest.update(data)
                        remaining -= len(data)
                        count += len(data)
                after = os.fstat(src.fileno())
            record.update(
                status="snapshot",
                bytes=count,
                initial_bytes=before.st_size,
                final_bytes=after.st_size,
                sha256=digest.hexdigest(),
                changed_during_copy=(before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns),
                complete_initial_extent=remaining == 0,
            )
        except (OSError, ValueError) as exc:
            record.update(status="unavailable", error=str(exc))
        record["finished_utc"] = utc_now()
        self.records.append(record)

    def read_text(self, name, source):
        # procfs/sysfs report size=0 despite having readable contents.
        try:
            value = Path(source).read_text(errors="replace")
            (self.output / name).write_text(value)
            self.records.append({"name": name, "source": str(source), "status": "ok"})
        except OSError as exc:
            self.records.append({"name": name, "source": str(source), "status": "unavailable", "error": str(exc)})

    def finish(self, args):
        files = {}
        for path in sorted(self.output.iterdir()):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                files[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        manifest = {
            "schema_version": 1,
            "captured_utc": utc_now(),
            "host": socket.gethostname(),
            "job_id": args.job_id,
            "pid": args.pid,
            "since": args.since,
            "records": self.records,
            "files": files,
            "scope": "read-only snapshot; missing evidence is not a passing health check",
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def collect(args):
    capture = Capture(args.output, args.timeout)
    commands = {
        "gpu-query": [
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,pci.bus_id,name,driver_version,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
            "--format=csv",
        ],
        "gpu-full": ["nvidia-smi", "-q"],
        "gpu-topology": ["nvidia-smi", "topo", "-m"],
        "gpu-processes": ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"],
        "kernel-monotonic": ["dmesg"],
        "kernel-wallclock": ["dmesg", "--time-format", "iso"],
        "kernel-journal": [
            "journalctl",
            "-k",
            "--since",
            args.since.replace("T", " ", 1),
            "--no-pager",
            "-o",
            "short-iso-precise",
        ],
        "cpu-cores": ["coredumpctl", "--since", args.since.replace("T", " ", 1), "--no-pager", "list"],
        "memory": ["free", "-b"],
        "disk": ["df", "-h", str(capture.output), "/tmp", "/dev/shm"],
        "disk-inodes": ["df", "-i", str(capture.output), "/tmp", "/dev/shm"],
    }
    if args.job_id:
        commands["slurm-job"] = ["scontrol", "show", "job", args.job_id]
        commands["slurm-accounting"] = [
            "sacct",
            "-j",
            args.job_id,
            "-S",
            args.since,
            "-P",
            "-o",
            "JobID,JobName,State,ExitCode,DerivedExitCode,Start,End,Elapsed,NodeList,MaxRSS,ReqMem,Reason",
        ]
    if args.python:
        commands["package-metadata"] = [args.python, "-c", METADATA_CODE]
    for name, argv in commands.items():
        capture.command(name, argv)
    for name, source in {
        "boot-id.txt": "/proc/sys/kernel/random/boot_id",
        "uptime.txt": "/proc/uptime",
        "core-pattern.txt": "/proc/sys/kernel/core_pattern",
        "driver.txt": "/proc/driver/nvidia/version",
        "collector-limits.txt": "/proc/self/limits",
    }.items():
        capture.read_text(name, source)
    for pattern in (
        "mc*/ce_count",
        "mc*/ue_count",
        "mc*/dimm*/dimm_ce_count",
        "mc*/dimm*/dimm_ue_count",
        "mc*/dimm*/dimm_label",
    ):
        for path in sorted(Path("/sys/devices/system/edac/mc").glob(pattern)):
            capture.read_text("edac-" + "-".join(path.parts[-3:]) + ".txt", path)
    if args.pid:
        proc = Path("/proc") / str(args.pid)
        for name in ("status", "limits", "cgroup", "stat"):
            capture.read_text("process-" + name + ".txt", proc / name)
        try:
            env = selected_environment((proc / "environ").read_bytes())
            (capture.output / "process-env-allowlist.json").write_text(json.dumps(env, indent=2) + "\n")
            maps = (proc / "maps").read_text().splitlines()
            libs = sorted(
                {
                    line.split()[-1]
                    for line in maps
                    if any(
                        key in line for key in ("libcuda", "libnccl", "libcudnn", "libcublas", "jaxlib", "cuda_plugin")
                    )
                }
            )
            (capture.output / "process-gpu-libraries.json").write_text(json.dumps(libs, indent=2) + "\n")
        except OSError as exc:
            capture.records.append({"name": "process-detail", "status": "unavailable", "error": str(exc)})
    for index, source in enumerate(args.log):
        capture.snapshot(f"input-{index:03d}-{Path(source).name}", source)
    capture.finish(args)
    return capture.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New private evidence directory; refuses reuse")
    parser.add_argument("--since", required=True, help="Incident window start, e.g. 2026-10-22T10:00:00")
    parser.add_argument("--job-id")
    parser.add_argument("--pid", type=int, help="Actual trainer PID, not a launcher PID")
    parser.add_argument("--python", help="Training prefix's interpreter; reads package metadata only")
    parser.add_argument("--log", type=Path, action="append", default=[], help="Explicit log/config file to preserve")
    parser.add_argument("--timeout", type=float, default=15, help="Seconds per read-only command")
    args = parser.parse_args()
    if args.timeout <= 0 or (args.pid is not None and args.pid <= 0):
        parser.error("timeout and PID must be positive")
    print(collect(args))


if __name__ == "__main__":
    main()
