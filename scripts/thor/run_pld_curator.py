"""Thor offline CPU probe/candidate/reward curation in the pinned image.

No network, NVIDIA runtime, hardware device, weight mount or inference restart.
Raw sessions and source are read-only; every run creates a separate output.
"""

import argparse
import datetime
import json
from pathlib import Path
import subprocess

IMAGE = "sha256:2711020dd277e9114ef6296ddc0ed721de376a4b1156183ed39c6c1450a2439c"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", type=Path, required=True, help="pinned source release containing packages/scripts/FK XML")
    p.add_argument("--raw-root", type=Path, required=True, help="complete copied sessions, including policy_events.jsonl")
    p.add_argument("--episodes", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True, help="NEW versioned run directory")
    p.add_argument("--export-evidence", action="store_true")
    p.add_argument("--purpose", choices=("probe", "candidates", "rewards"), default="probe")
    p.add_argument("--rewards", type=Path, help="reviewed immutable JSON, mounted read-only")
    p.add_argument("--validation-episodes", type=Path, nargs="*", default=[])
    args = p.parse_args()
    code, raw, output = (path.resolve() for path in (args.code, args.raw_root, args.output))
    for member in ("scripts/pld_rl/curate_recording.py", "packages/pld-rl/src/yam_pld_rl/native_recording.py",
                   "third_party/yam-fk-model/yam_linear4310.xml"):
        if not (code / member).is_file():
            p.error(f"missing release member: {member}")
    if output.is_relative_to(raw) or output.is_relative_to(code):
        p.error("output must be outside immutable raw/source trees")
    episodes = [ep.resolve() for ep in args.episodes]
    if not episodes or any(not ep.is_relative_to(raw) or not (ep / "manifest.json").is_file() for ep in episodes):
        p.error("episodes must be finalized native recordings below --raw-root")
    validation = [ep.resolve() for ep in args.validation_episodes]
    if any(ep not in episodes for ep in validation):
        p.error("validation episodes must be in this exact input list")
    if (args.purpose == "rewards") != (args.rewards is not None):
        p.error("--rewards is required only with --purpose rewards")
    review = args.rewards.resolve() if args.rewards else None
    if review and not review.is_file():
        p.error("review file does not exist")
    actual_image = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if actual_image != IMAGE:
        raise ValueError("offline image identity differs")
    output.mkdir(parents=True, exist_ok=False)
    name = "pld-curator-" + args.purpose + "-" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%f")
    # getuid/gid are the data owner, not a repurposed shell/system variable.
    import os  # noqa: PLC0415

    argv = ["docker", "run", "--name", name, "--restart", "no", "--runtime", "runc", "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only", "--cpus", "2", "--memory", "4g",
            "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m",
            "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
            "-v", f"{code}:/workspace:ro", "-v", f"{raw}:/raw:ro", "-v", f"{output}:/output:rw"]
    if review:
        argv += ["-v", f"{review}:/review.json:ro"]
    argv += ["--entrypoint", "/opt/parts-vision/bin/python", IMAGE, "/workspace/scripts/pld_rl/curate_recording.py",
             "--episodes", *[str(Path("/raw") / ep.relative_to(raw)) for ep in episodes], "--output", "/output/curator"]
    if args.purpose == "probe":
        argv.append("--probe-audit")
    if review:
        argv += ["--rewards", "/review.json"]
    if validation:
        argv += ["--validation-episodes", *[str(Path("/raw") / ep.relative_to(raw)) for ep in validation]]
    if args.export_evidence:
        argv.append("--export-evidence")
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    (output / "stdout.log").write_text(result.stdout)
    (output / "stderr.log").write_text(result.stderr)
    receipt = {"container": name, "image_id": IMAGE, "argv": argv, "returncode": result.returncode, "purpose": args.purpose,
               "source_release": str(code), "source_commit": (code / "source-commit.txt").read_text().strip()
               if (code / "source-commit.txt").is_file() else None,
               "training": False, "hardware_io": False, "gpu": False,
               "production_inference_modified": False, "raw_modified": False}
    (output / "launcher.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"container": name, "returncode": result.returncode, "output": str(output)}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
