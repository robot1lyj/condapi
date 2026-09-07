"""Launch recorded precision runs on Thor, restoring daily power after each run."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MODES = {"A": ("checkpoint", "float32"), "B": ("checkpoint", "bfloat16"), "C": ("bfloat16", "bfloat16")}
TORCH_MODES = {
    "D": ("float32", "float32"),
    "E": ("bfloat16", "bfloat16"),
    "F": ("bfloat16", "bfloat16"),
    "G": ("bfloat16", "bfloat16"),
    "H": ("bfloat16", "bfloat16"),
    "I": ("bfloat16", "bfloat16"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/wuyan-lyj/thor/pi"))
    parser.add_argument("--image", default="openpi-pi:thor-jax-candidate")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--modes", nargs="+", choices=(*MODES, *TORCH_MODES), default=list(MODES))
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Run with sudo on Thor")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.batch_id):
        parser.error("batch-id must be a filename-safe identifier")
    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[1]
    for mode in args.modes:
        label = f"pi05-{mode}-{args.batch_id}"
        prefix = args.root / "logs" / label
        is_pytorch = mode in TORCH_MODES
        params, compute = (TORCH_MODES if is_pytorch else MODES)[mode]
        checkpoint = "pi05_base_pytorch_fp32_v1" if is_pytorch else "pi05_base"
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
            "-e",
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor",
            "-e",
            "TRITON_CACHE_DIR=/cache/triton",
            "-v",
            f"{scripts}:/bench:ro",
            "-v",
            f"{args.root / 'checkpoints'}:/checkpoints:ro",
            "-v",
            f"{args.root / 'cache'}:/cache",
            "-v",
            f"{args.root / 'test-data'}:/test-data:ro",
            "-v",
            f"{args.root / 'results'}:/results",
            args.image,
            "python",
            "/bench/benchmark_suite.py",
            "--checkpoint",
            f"/checkpoints/{checkpoint}",
            "--suite",
            "/test-data/pi05-replay-v1/suite.json",
            "--params-dtype",
            params,
            "--compute-dtype",
            compute,
            "--output",
            f"/results/{label}",
        ]
        if is_pytorch:
            command.extend(["--backend", "pytorch", "--compile" if mode in ("F", "G", "H", "I") else "--no-compile"])
            if mode in ("F", "G", "H", "I"):
                command.append("--native-attention-mask")
            if mode in ("G", "H"):
                command.extend(["--attention", "sdpa"])
            if mode in ("H", "I"):
                command.append("--batch-vision")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        files = sorted([*scripts.glob("*.py"), *repo.glob("src/openpi/**/*.py"), *repo.glob("scripts/docker/thor/*")])
        hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
        manifest = {
            "run_id": label,
            "phase": "evaluate",
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "code_commit_base": args.code_commit,
            "source_files_sha256": hashes,
            "source_state": "working_tree_fingerprinted; image is authoritative for installed model code",
            "image_id": subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
            ).strip(),
            "hardware": Path("/proc/device-tree/model").read_text().rstrip("\x00"),
            "kernel": os.uname().release,
            "command": command,
            "power_before": subprocess.check_output(["nvpmodel", "-q"], text=True),
            "checkpoint": f"{checkpoint}; no LoRA; read-only checkpoint",
            "network": "disabled for inference container; all inputs local",
        }
        with prefix.with_suffix(".manifest.json").open("x") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
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
        with prefix.with_suffix(".power-after.json").open("x") as output:
            json.dump(
                {"exit_code": code, "power_after": subprocess.check_output(["nvpmodel", "-q"], text=True)}, output
            )
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
