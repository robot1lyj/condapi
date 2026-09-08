"""Local Conda execution with immutable run directories and explicit provenance."""

from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from vla_platform.contracts import require
from vla_platform.project import digest
from vla_platform.project import write_json


def now():
    return datetime.now(UTC).isoformat()


def git_state(root):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Include untracked files in provenance without recording their contents.
    tracked = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    import hashlib  # noqa: PLC0415

    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root)
    files = {name: digest(Path(root) / name) for name in untracked.decode().split("\0") if name}
    return {
        "commit": head,
        "tracked_patch_sha256": hashlib.sha256(tracked).hexdigest(),
        "untracked_sha256": files,
        "dirty": bool(tracked or files),
    }


def execute(plan):
    require(
        plan.operation != "train" or os.environ.get("SLURM_JOB_ID") or os.environ.get("TMUX"),
        "Training must run inside a Slurm allocation or tmux on an audited compute node",
    )
    require((Path(plan.prefix) / "conda-meta").is_dir(), "Conda environment has not been created")
    for path, expected in plan.source_hashes.items():
        require(digest(path) == expected, f"Configuration/source changed since planning: {path}")
    command = list(plan.command)
    if plan.target == "thor":
        device = Path("/proc/device-tree/model")
        require(device.is_file() and "Thor" in device.read_text(), "Thor profile must execute on the Thor host")
        if plan.operation in ("infer", "benchmark"):
            conda = shutil.which("conda")
            require(conda is not None, "Conda executable not found")
            command[0] = conda
            command = ["sudo", "/usr/bin/python3", str(Path(plan.cwd) / "scripts/thor/maxn_session.py"), "--", *command]
    output = Path(plan.output)
    output.mkdir(parents=True, exist_ok=False)
    state = git_state(plan.cwd)
    write_json(output / "plan.json", plan.to_dict())
    write_json(
        output / "started.json",
        {
            "started_at": now(),
            "git": state,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "effective_command": command,
            "status": "running",
        },
    )
    child = None
    exit_code = None
    error = None
    previous = {}

    def interrupt(signum, _frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        with (output / "console.log").open("x") as log:
            child = subprocess.Popen(
                command, cwd=plan.cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            exit_code = child.wait()
    except (OSError, KeyboardInterrupt) as exc:
        error = type(exc).__name__
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 127
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write_json(
            output / "finished.json",
            {
                "finished_at": now(),
                "exit_code": exit_code,
                "error": error,
                "status": "command_succeeded_not_model_accepted" if exit_code == 0 else "failed",
            },
        )
    return exit_code


def audit_environment(prefix):
    require(Path(prefix).is_absolute(), "Expected absolute prefix")
    result = subprocess.run(
        ["conda", "list", "--prefix", str(prefix), "--json"], capture_output=True, text=True, check=True
    )
    return {
        "observed_at": now(),
        "prefix": str(prefix),
        "packages": json.loads(result.stdout),
        "status": "inventory_only_not_gpu_validated",
    }
