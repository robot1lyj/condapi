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
    parser.add_argument("--stage", choices=("export", "engine", "profile"), default="export")
    parser.add_argument("--source-export")
    parser.add_argument("--source-engine")
    parser.add_argument("--cache-time-modulation", action="store_true")
    parser.add_argument("--text-bucket", type=int, choices=(80, 128, 200), default=200)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/home/wuyan-lyj/thor/pi"))
    args = parser.parse_args()
    if os.geteuid() != 0 or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        parser.error("Run with sudo and a filename-safe run-id")
    if args.stage == "engine" and (not args.source_export or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.source_export)):
        parser.error("Engine build requires a filename-safe source-export ID")
    if args.stage == "profile" and (not args.source_engine or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.source_engine)):
        parser.error("Profiling requires a filename-safe source-engine ID")
    if args.cache_time_modulation and args.stage != "export":
        parser.error("Time cache is an export preparation option")
    if args.stage != "export" and (args.prepare_only or args.text_bucket != 200 or args.compute_dtype != "bfloat16"):
        parser.error("Preparation options require export stage")
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
    if args.cache_time_modulation:
        command.append("--cache-time-modulation")
    if args.stage == "export":
        command.extend(["--text-bucket", str(args.text_bucket), "--compute-dtype", args.compute_dtype])
        if args.prepare_only:
            command.append("--prepare-only")
    if args.stage == "engine":
        command = [
            *command[: command.index("/bench/export_pi05_onnx.py")],
            "/bench/build_trt_engine.py",
            "--source",
            f"/artifacts/{args.source_export}",
            "--output",
            f"/artifacts/{args.run_id}",
        ]
    if args.stage == "profile":
        command = [
            *command[: command.index("/bench/export_pi05_onnx.py")],
            "/bench/profile_trt.py",
            "--engine",
            f"/artifacts/{args.source_engine}",
            "--suite",
            "/test-data/pi05-replay-v1/suite.json",
            "--output",
            f"/artifacts/{args.run_id}",
        ]
    manifest = {
        "run_id": args.run_id,
        "phase": args.stage,
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "code_commit_base": args.code_commit,
        "source_files_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "image_id": subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
        ).strip(),
        "command": command,
        "power_before": subprocess.check_output(["nvpmodel", "-q"], text=True),
        "scope": "GPU reference/preparation inference, trace or engine tactic profiling; no input overwrite, network or robot IO",
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
