"""Verify the approved specialist/replay contract and call native OpenPI.

Standard-library control wrapper; model and data validation run as dedicated
subprocesses on an authorized Slurm compute node, never on the workstation.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "configs/native/pi05-lego-separation-r1.json"
RECIPE_10K = ROOT / "configs/native/pi05-lego-separation-r1-10k.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_recipe(path):
    recipe = read_json(path)
    report = ROOT / recipe["report"] / "bimanual100-replay100"
    release, dataset = read_json(report / "release.json"), read_json(report / "dataset.json")
    audit = read_json(report / "audit-summary.json")
    identities = release["episode_lists"]
    for name, expected in identities.items():
        if sha256(report / name) != expected:
            raise ValueError(f"Frozen episode list changed: {name}")
    if sha256(report / "episodes.json") != recipe["selection_sha256"]:
        raise ValueError("Approved 200-episode selection changed")
    if sha256(report / "dataset.json") != release["dataset_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    if sha256(report / "audit.jsonl") != release["audit_jsonl_sha256"] or audit["status"] != "passed":
        raise ValueError("Missing complete data audit")
    pools = [set(read_json(report / name)) for name in ("specialist-episodes.json", "replay-episodes.json")]
    if (
        any(len(pool) != 100 for pool in pools)
        or pools[0] & pools[1]
        or pools[0] | pools[1] != set(read_json(report / "episodes.json"))
    ):
        raise ValueError("Requires disjoint specialist100/replay100")
    if dataset["source"] != recipe["dataset_root"] or dataset["validation_split_included"] is not False:
        raise ValueError("Dataset source or split changed")
    if any(stats["frames"] != 278516 for stats in dataset["pool_stats"].values()):
        raise ValueError("Approved 50/50 frame mixture changed")
    approved_budgets = {"recipe-draft.md v1": 5000, "recipe-v2-10k": 10000}
    version = recipe["approval"]["recipe_version"]
    if version not in approved_budgets:
        raise ValueError("Unknown confirmed recipe version")
    budget = approved_budgets[version]
    expected = {
        "config": "pi05_yam",
        "batch_size": 32,
        "fsdp_devices": 4,
        "num_workers": 2,
        "total_steps": budget,
        "horizon": 50,
        "rtc_training_max_delay": 10,
        "peak_lr": 3e-6,
        "decay_lr": 3e-7,
        "warmup_steps": 200,
        "decay_steps": budget,
        "save_interval": 1000,
        "keep_period": 1000,
        "seed": 42,
        "ema_decay": None,
        "adam_eps": 1e-6,
    }
    if (
        recipe["training"] != expected
        or recipe["automatic_training_start"] is not False
        or recipe["approval"]["training_authorized"] is not True
    ):
        raise ValueError("Recipe differs from explicitly confirmed version")
    return recipe, report, dataset


def check_server_data(recipe, report, dataset, *, full_hashes):
    source, selected = Path(recipe["dataset_root"]), Path(recipe["selection_root"])
    parent = Path(recipe["parent_checkpoint"])
    for name in ("episodes.json", "specialist-episodes.json", "replay-episodes.json", "dataset.json"):
        if sha256(selected / name) != sha256(report / name):
            raise ValueError(f"Server selection differs from frozen report: {name}")
    for name, field in (
        ("meta/info.json", "source_info_sha256"),
        ("conversion_manifest.json", "source_manifest_sha256"),
    ):
        if sha256(source / name) != dataset[field]:
            raise ValueError("Source metadata identity changed")
    if not (parent / "_CHECKPOINT_METADATA").is_file() or not (parent / "train_state/_METADATA").is_file():
        raise ValueError("Parent checkpoint missing committed state")
    if sha256(parent / "params/_METADATA") != recipe["parent_metadata_sha256"]:
        raise ValueError("Parent parameter metadata changed")
    if sha256(parent / "assets/yam/norm_stats.json") != recipe["parent_norm_sha256"]:
        raise ValueError("Inherited norm changed")
    audits = {
        r["episode_index"]: r for r in (json.loads(line) for line in (report / "audit.jsonl").read_text().splitlines())
    }
    for item in dataset["episodes"]:
        proof = audits[item["episode_index"]]
        if full_hashes and sha256(item["data_path"]) != proof["data_sha256"]:
            raise ValueError("Audited numeric data changed")
        for role, reference in item["videos"].items():
            path = Path(reference["path"])
            if (path.stat().st_size, path.stat().st_mtime_ns) != (reference["size"], reference["mtime_ns"]):
                raise ValueError("Audited video signature changed")
            if full_hashes and sha256(path) != proof["videos"][role]["sha256"]:
                raise ValueError("Audited video content changed")


def train_command(recipe, *, resume=False):
    training = recipe["training"]
    command = [
        recipe["python"],
        str(ROOT / "scripts/train_lego_full.py"),
        "--run-name",
        recipe["run_name"],
        "--steps",
        str(training["total_steps"]),
        "--init-params",
        str(Path(recipe["parent_checkpoint"]) / "params"),
        "--train-episodes-file",
        str(Path(recipe["selection_root"]) / "episodes.json"),
        "--rtc-training-max-delay",
        str(training["rtc_training_max_delay"]),
        "--warmup-steps",
        str(training["warmup_steps"]),
        "--decay-steps",
        str(training["decay_steps"]),
    ]
    if resume:
        command.append("--resume")
    environment = dict(os.environ)
    for key in ("batch_size", "num_workers", "peak_lr", "decay_lr", "save_interval", "keep_period"):
        environment[f"LEGO_{key.upper()}"] = str(training[key])
    environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts'}:{ROOT / 'packages/openpi-client/src'}"
    environment.pop("JAX_PLATFORMS", None)
    return command, environment


def launch(recipe, report, dataset, *, resume=False):
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not Path(recipe["python"]).is_file()
        or not Path(recipe["dataset_root"]).is_dir()
    ):
        raise RuntimeError("Requires the authorized server Slurm allocation")
    control = Path(recipe["control_root"])
    control.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "recipe": recipe,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "entry_sha256": sha256(ROOT / "scripts/train_lego_full.py"),
        "selection_sha256": recipe["selection_sha256"],
    }
    with (control / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = control / "launch_fingerprint.json"
        if resume:
            if read_json(frozen) != fingerprint:
                raise ValueError("Cannot resume changed recipe/code")
            check_server_data(recipe, report, dataset, full_hashes=False)
        else:
            if frozen.exists():
                raise FileExistsError("Existing launch attempt: inspect it before restarting")
            check_server_data(recipe, report, dataset, full_hashes=True)
            command, environment = train_command(recipe)
            cpu_environment = dict(environment, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
            preflight = control / "preflight"
            preflight.mkdir(exist_ok=False)
            with (preflight / "checkpoint-finite.log").open("x") as log:
                subprocess.run(
                    [
                        recipe["python"],
                        str(ROOT / "scripts/thor/audit_checkpoint_finite.py"),
                        "--checkpoint",
                        recipe["parent_checkpoint"],
                        "--output",
                        str(preflight / "checkpoint-finite.json"),
                    ],
                    env=cpu_environment,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if read_json(preflight / "checkpoint-finite.json")["nonfinite_elements"] != 0:
                raise ValueError("Nonfinite parent weights")
            with (preflight / "loader.log").open("x") as log:
                subprocess.run(
                    [
                        recipe["python"],
                        str(ROOT / "scripts/probe_yam_loader.py"),
                        "--dataset",
                        recipe["dataset_root"],
                        "--selection",
                        str(Path(recipe["selection_root"]) / "episodes.json"),
                        "--assets-root",
                        str(Path(recipe["parent_checkpoint"]) / "assets"),
                        "--all-selection",
                        "--output",
                        str(preflight / "loader"),
                    ],
                    env=cpu_environment,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            loader = read_json(preflight / "loader/summary.json")
            if loader["samples_failed"] or loader["samples_ok"] != 800 or loader["probe_frames"] != 557032:
                raise ValueError("Native loader preflight failed")
            with frozen.open("x") as handle:
                json.dump(fingerprint, handle, indent=2)
            print("PREFLIGHT_PASSED checkpoint_finite=0 loader_samples=800 source_hashes=verified", flush=True)
        command, environment = train_command(recipe, resume=resume)
        print("NATIVE_TRAIN_START", recipe["run_name"], "job=" + os.environ["SLURM_JOB_ID"], flush=True)
        with (control / "train.log").open("a") as log:
            subprocess.run(command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("action", choices=["check", "train"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    recipe, report, dataset = checked_recipe(args.recipe)
    if args.action == "check":
        print(
            json.dumps(
                {
                    "status": "APPROVED_RECIPE_AND_FROZEN_AUDIT_VERIFIED",
                    "run_name": recipe["run_name"],
                    "pool_stats": dataset["pool_stats"],
                }
            )
        )
    else:
        launch(recipe, report, dataset, resume=args.resume)


if __name__ == "__main__":
    main()
