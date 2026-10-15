"""Pinned Thor batch curation/judging/publication with immutable raw mounts."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from run_pld_training import offline_gpu

CPU_IMAGE = "sha256:2711020dd277e9114ef6296ddc0ed721de376a4b1156183ed39c6c1450a2439c"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--vision-image", required=True)
    p.add_argument("--stage", choices=("curate", "judge", "publish"), required=True)
    args = p.parse_args()
    if platform.machine() != "aarch64":
        p.error("Thor only")
    root = args.dataset.resolve()
    refs, models = args.references.resolve(), args.models.resolve()
    code = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=code, text=True).strip():
        p.error("clean pinned checkout required")
    audit = root / (args.stage + "-pipeline-v1")
    audit.mkdir(exist_ok=False)
    statuses = []

    def step(name, script, argv, *, gpu=False, image=CPU_IMAGE):
        if gpu:
            offline_gpu()
        command = ["docker", "run", "--name", "pld-closing8-" + name + "-r06",
                   "--restart", "no", "--network", "none", "--read-only", "--cpus", "4",
                   "--memory", "24g" if gpu else "8g", "--shm-size", "1g",
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
                   "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OPENBLAS_NUM_THREADS=1",
                   "-e", "OMP_NUM_THREADS=2", "-e", "HF_HOME=/tmp/hf", "-e", "XDG_CACHE_HOME=/tmp/cache",
                   "-v", f"{code}:/workspace:ro", "-v", f"{root}:{root}:rw",
                   "-v", f"{root / 'raw-v1'}:{root / 'raw-v1'}:ro",
                   "-v", f"{refs}:{refs}:ro", "-v", f"{models}:/models:ro"]
        command += ["--runtime", "nvidia" if gpu else "runc", "-e",
                    "NVIDIA_VISIBLE_DEVICES=all" if gpu else "NVIDIA_VISIBLE_DEVICES=void"]
        command += ["--entrypoint", "/opt/parts-vision/bin/python", image,
                    "/workspace/" + script, *[str(a) for a in argv]]
        rec = {"stage": name, "argv": command, "started_unix": time.time(), "status": "running"}
        statuses.append(rec)
        (audit / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
        with (audit / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        rec.update(returncode=result.returncode, ended_unix=time.time(),
                   status="complete" if result.returncode == 0 else "failed")
        (audit / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
        print(json.dumps({"stage": name, "returncode": result.returncode}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    vision, intent = root / "model-inputs-v1/vision-v1", root / "model-inputs-v1/intent-v1"
    if args.stage == "curate":
        snap = json.loads((root / "source-snapshot-v1.json").read_text())
        raw = root / "raw-v1" / Path(snap["source_root"]).name
        episodes = [raw / e["path"] for e in snap["episodes"]]
        step("curate", "scripts/pld_rl/curate_recording.py",
             ["--episodes", *episodes, "--validation-episodes", episodes[0],
              "--export-evidence", "--output", root / "curator-v1"])
        step("stage", "scripts/pld_rl/prepare_reward_inputs.py",
             ["--curator", root / "curator-v1", "--references", refs, "--output", root / "model-inputs-v1"])
        step("crops", "scripts/parts_rl/prepare_revised_features.py",
             ["--dataset", vision, "--references", vision / "reference-labels-staged.json"])
    elif args.stage == "judge":
        step("dino", "scripts/parts_rl/probe_dinov3.py",
             ["--dataset", vision, "--weights", "/models/timm-dinov3-vits16-3bf4720/model.safetensors",
              "--skip-dense"], gpu=True, image=args.vision_image)
        step("qwen", "scripts/parts_rl/review_grasps_vlm.py",
             ["--dataset", vision, "--model", "/models/Qwen3-VL-4B-Instruct-ebb281ec",
              "--batch-size", "8", "--terminal-hold"], gpu=True, image=args.vision_image)
        step("intent", "scripts/parts_rl/review_grasp_intent.py",
             ["--dataset", intent, "--model", "/models/Qwen3-VL-4B-Instruct-ebb281ec",
              "--batch-size", "8"], gpu=True, image=args.vision_image)
        step("classify", "scripts/parts_rl/classify_revised_grasps.py", ["--dataset", vision])
    else:
        step("merge", "scripts/pld_rl/merge_reward_evidence.py",
             ["--curator", root / "curator-v1", "--vision", vision, "--intent", intent,
              "--output", root / "vision-combined-v1"])
        step("rewards", "scripts/pld_rl/publish_collect_rewards.py",
             ["--dataset", root, "--curator", "curator-v1", "--vision", "vision-combined-v1",
              "--output", root / "reward-release-v1"])
        step("replay", "scripts/pld_rl/prepare_blocks.py",
             ["--curator", root / "reward-release-v1", "--bundles", root / "behavior-bundle",
              "--output", root / "replay-v1"])
        step("learning", "scripts/pld_rl/prepare_learning_view.py",
             ["--replay", root / "replay-v1", "--output", root / "learning-replay-v1"])
    (audit / "complete.json").write_text(json.dumps({"status": "complete", "source": source,
        "stage": args.stage, "raw_modified": False, "tracking": False, "training": False}) + "\n")


if __name__ == "__main__":
    main()
