"""Record a Thor GPU ONNX export/preparation run, including MAXN cleanup."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/wuyan-lyj/thor/pi"))
    args = parser.parse_args()
    if os.geteuid() != 0 or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        parser.error("Run with sudo and a filename-safe run-id")
    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[1]
    artifacts = args.root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    prefix = args.root / "logs" / args.run_id
    prefix.parent.mkdir(exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        args.run_id,
        "--runtime=nvidia",
        "--gpus",
        "all",
        "--network",
        "none",
        "--shm-size=8g",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-v",
        f"{scripts}:/bench:ro",
        "-v",
        f"{args.root / 'checkpoints'}:/checkpoints:ro",
        "-v",
        f"{args.root / 'test-data'}:/test-data:ro",
        "-v",
        f"{args.root / 'cache'}:/cache",
        "-v",
        f"{artifacts}:/artifacts",
        args.image,
        "python",
        "/bench/export_pi05_onnx.py",
        "--checkpoint",
        "/checkpoints/pi05_base_pytorch_fp32_v1",
        "--suite",
        "/test-data/pi05-replay-v1/suite.json",
        "--output",
        f"/artifacts/{args.run_id}",
    ]
    files = sorted([*scripts.glob("*.py"), *repo.glob("src/openpi/**/*.py")])
    manifest = {
        "run_id": args.run_id,
        "phase": "export",
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "code_commit_base": args.code_commit,
        "source_files_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "image_id": subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
        ).strip(),
        "command": command,
        "power_before": subprocess.check_output(["nvpmodel", "-q"], text=True),
        "scope": "GPU reference/preparation inference and trace; no weight overwrite, network or robot IO",
    }
    with prefix.with_suffix(".manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    code = subprocess.call(
        [
            sys.executable,
            str(scripts / "maxn_session.py"),
            "--",
            sys.executable,
            str(scripts / "logged_command.py"),
            "--prefix",
            str(prefix),
            "--",
            *command,
        ]
    )
    with prefix.with_suffix(".power-after.json").open("x") as stream:
        json.dump({"exit_code": code, "power_after": subprocess.check_output(["nvpmodel", "-q"], text=True)}, stream)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
