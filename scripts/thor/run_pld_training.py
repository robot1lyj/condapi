"""Prepare an isolated Thor RL worker or run an explicit offline SAC job."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import subprocess

NAME = "pld-rl-train"


def inspect(name):
    return json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]


def offline_gpu():
    ids = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
    for cid in ids:
        item = inspect(cid)
        if item["Name"] == "/" + NAME:
            continue
        env = dict(v.split("=", 1) for v in item["Config"].get("Env", []) if "=" in v)
        visible = env.get("NVIDIA_VISIBLE_DEVICES", "void")
        if (item["HostConfig"].get("DeviceRequests")
                or (item["HostConfig"].get("Runtime") == "nvidia" and visible not in ("", "void", "none"))):
            raise RuntimeError(f"stop/wait for GPU workload before training: {item['Name']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "validate", "train"))
    p.add_argument("--code", type=Path, required=True, help="clean, pinned standalone Git checkout")
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--image", required=True, help="built immutable sha256 image ID")
    p.add_argument("--run-name", help="new run directory, never overwrite")
    p.add_argument("--config", type=Path)
    p.add_argument("--contract", type=Path)
    p.add_argument("--normalization", type=Path)
    p.add_argument("--replay", type=Path)
    p.add_argument("--validation-suite", choices=("grasp-v3", "calibrated"), default="grasp-v3")
    args = p.parse_args()
    if platform.machine() != "aarch64":
        p.error("This launcher executes on Thor only")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        p.error("pin the full built image ID")
    code, datasets, runs, encoder = [x.resolve() for x in (args.code, args.datasets, args.runs, args.encoder)]
    if not code.is_dir() or not datasets.is_dir() or not encoder.is_file():
        p.error("source/dataset/encoder missing")
    if runs.is_relative_to(code) or runs.is_relative_to(datasets):
        p.error("runs must be outside read-only source and datasets")
    runs.mkdir(parents=True, exist_ok=True)
    expected = {str(code): "/workspace", str(datasets): "/datasets", str(runs): "/runs", str(encoder): "/encoder/resnet18.pth"}
    if args.action == "prepare":
        argv = ["docker", "run", "-d", "--name", NAME, "--restart", "no", "--runtime", "nvidia", "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--cpus", "6", "--memory", "16g",
                "--shm-size", "1g", "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
                "-e", "NVIDIA_VISIBLE_DEVICES=all", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                "-e", "XDG_CACHE_HOME=/tmp/rl-cache"]
        for source, target in expected.items():
            argv += ["-v", f"{source}:{target}:{'rw' if target == '/runs' else 'ro'}"]
        argv.append(args.image)
        subprocess.run(argv, check=True)
        return
    worker = inspect(NAME)
    mounts = {m["Source"]: m["Destination"] for m in worker["Mounts"] if m["Type"] == "bind"}
    if worker["Image"] != args.image or mounts != expected or not worker["State"]["Running"]:
        p.error("worker image/mounts/state differ from the requested pinned run")
    if any(m["RW"] != (m["Destination"] == "/runs") for m in worker["Mounts"] if m["Type"] == "bind"):
        p.error("worker source/datasets/encoder must be read-only")
    lock = (runs / ".pld-rl-train.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("another RL job already owns this training worker")
    offline_gpu()
    if not args.run_name or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}", args.run_name):
        p.error("provide a simple new --run-name")
    output = runs / args.run_name
    if output.exists():
        p.error("run directory already exists; resume is a separate explicit operation")

    def inside(path):
        if path is None or not path.exists():
            p.error("training requires existing config/contract/normalization/replay")
        resolved = path.resolve()
        for host, target in ((code, Path("/workspace")), (datasets, Path("/datasets"))):
            if resolved.is_relative_to(host):
                return str(target / resolved.relative_to(host))
        raise ValueError("training inputs must be in the read-only source or dataset mount")

    argv = ["docker", "exec", NAME, "/opt/pld-rl/bin/python", "-u"]
    if args.action == "validate":
        script = "calibrated_gpu_checks.py" if args.validation_suite == "calibrated" else "validate_grasp_v3_gpu.py"
        argv += ["/workspace/scripts/pld_rl/" + script]
    else:
        argv += ["/workspace/scripts/pld_rl/train_sac.py", "--config", inside(args.config),
                 "--contract", inside(args.contract), "--normalization", inside(args.normalization), "--replay", inside(args.replay)]
    argv += ["--execution", "thor-offline", "--encoder", "/encoder/resnet18.pth", "--output", "/runs/" + args.run_name]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    # A failed startup may not have created the learner output; retain its diagnostics separately.
    logs = runs / (args.run_name + ".launcher")
    logs.mkdir(exist_ok=False)
    (logs / "stdout.log").write_text(result.stdout)
    (logs / "stderr.log").write_text(result.stderr)
    (logs / "launcher.json").write_text(json.dumps({"action": args.action, "container": NAME, "image": args.image,
        "argv": argv, "returncode": result.returncode, "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code, text=True).strip(), "hardware_io": False,
        "production_weights_modified": False}, indent=2) + "\n")
    print(json.dumps({"returncode": result.returncode, "run": str(output), "logs": str(logs)}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
