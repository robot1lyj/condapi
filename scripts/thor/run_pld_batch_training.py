"""Sequential isolated Thor experiments and checkpoint/export verification."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    code = Path(__file__).resolve().parents[2]
    data = args.dataset.resolve()
    datasets = data.parent
    runs = datasets.parent / "runs"
    audit = runs / args.run_name
    audit.mkdir(exist_ok=False)
    jobs = json.loads((data / "training-input-v1/summary.json").read_text())["jobs"]
    statuses = []
    for name in jobs:
        cmd = ["python3", str(code / "scripts/thor/run_pld_training.py"), "train",
               "--code", str(code), "--datasets", str(datasets), "--runs", str(runs),
               "--encoder", str(args.encoder), "--image", args.image, "--run-name", name,
               "--config", str(data / "training-input-v1/configs" / (name + ".json")),
               "--contract", str(data / "training-input-v1/contract.json"),
               "--normalization", str(data / "training-input-v1/normalization.json"),
               "--replay", str(data / "learning-replay-v1")]
        rec = {"run": name, "argv": cmd, "started_unix": time.time(), "status": "running"}
        statuses.append(rec)
        (audit / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
        with (audit / (name + ".log")).open("w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        rec.update(returncode=r.returncode, ended_unix=time.time(),
                   status="complete" if r.returncode == 0 else "failed")
        (audit / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
        if r.returncode:
            raise SystemExit(r.returncode)
        check = ["docker", "exec", "pld-rl-train", "/opt/pld-rl/bin/python",
                 "/workspace/scripts/pld_rl/audit_calibrated_run.py", "--run", "/runs/" + name,
                 "--replay", "/datasets/" + data.name + "/learning-replay-v1",
                 "--output", "/runs/" + args.run_name + "/" + name + "-audit.json"]
        with (audit / (name + "-audit.log")).open("w") as f:
            r = subprocess.run(check, stdout=f, stderr=subprocess.STDOUT, check=False)
        rec["audit_returncode"] = r.returncode
        (audit / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
        print(json.dumps(rec), flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)
    (audit / "complete.json").write_text(json.dumps({"status": "complete", "runs": jobs,
        "production_actor_replaced": False, "new_environment_interactions": 0}) + "\n")


if __name__ == "__main__":
    main()
