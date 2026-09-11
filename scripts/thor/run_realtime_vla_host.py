"""Run a separately recorded Dexmal replay, restoring daily Thor power on exit."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def docker_command(root, scripts, *, image, label, position_offset=-1, fix_time_broadcast=False, text_capacity=200):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Unsafe run label")
    if position_offset not in (-1, 0) or text_capacity not in (80, 200):
        raise ValueError("Unsupported experiment")
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        label,
        "--runtime=nvidia",
        "--gpus",
        "all",
        "--network",
        "none",
        "--shm-size=8g",
    ]
    for value in (
        "JAX_PLATFORMS=cpu",
        "PYTHONUNBUFFERED=1",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0",
        "OPENPI_DATA_HOME=/pi/cache",
        "TRITON_CACHE_DIR=/pi/cache/realtime-vla-triton",
        "TORCHINDUCTOR_CACHE_DIR=/pi/cache/realtime-vla-inductor",
        "PYTHONPATH=/bench",
    ):
        command.extend(["-e", value])
    for mount in (
        f"{root}:/pi:ro",
        f"{root / 'results'}:/pi/results",
        f"{root / 'cache'}:/pi/cache",
        f"{scripts}:/bench:ro",
    ):
        command.extend(["-v", mount])
    command.extend(
        [
            image,
            "python",
            "/bench/benchmark_realtime_vla.py",
            "--source",
            "/pi/references/realtime-vla-b86a942",
            "--bundle",
            "/pi/artifacts/realtime-vla-b86a942-20260911",
            "--suite",
            "/pi/test-data/pi05-replay-v1/suite.json",
            "--reference",
            "/pi/results/pi05-A-20260907-r2",
            "--output",
            f"/pi/results/{label}",
            "--position-offset",
            str(position_offset),
            "--text-capacity",
            str(text_capacity),
        ]
    )
    if fix_time_broadcast:
        command.append("--fix-time-broadcast")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/wuyan-lyj/thor/pi"))
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--control-scripts", type=Path, default=Path("/home/wuyan-lyj/condapi/scripts/thor"))
    parser.add_argument("--image", default="openpi-pi:thor-pytorch-onnx-v6-20260907")
    parser.add_argument("--label", required=True)
    parser.add_argument("--position-offset", type=int, choices=(-1, 0), default=-1)
    parser.add_argument("--fix-time-broadcast", action="store_true")
    parser.add_argument("--text-capacity", type=int, choices=(80, 200), default=200)
    args = parser.parse_args()
    command = docker_command(
        args.root,
        args.scripts,
        image=args.image,
        label=args.label,
        position_offset=args.position_offset,
        fix_time_broadcast=args.fix_time_broadcast,
        text_capacity=args.text_capacity,
    )
    prefix = args.root / "logs" / args.label
    manifest = {
        "command": command,
        "image_id": subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
        ).strip(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.scripts.glob("*.py")},
        "power_before": subprocess.check_output(["nvpmodel", "-q"], text=True),
    }
    with prefix.with_suffix(".manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    code = subprocess.call(
        [
            sys.executable,
            str(args.control_scripts / "maxn_session.py"),
            "--",
            sys.executable,
            str(args.control_scripts / "logged_command.py"),
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
