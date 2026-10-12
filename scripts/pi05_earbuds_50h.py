"""Prepare/check the earbuds subset, or explicitly launch the existing OpenPI entrypoint.

No default action and no training imports. Data preparation never starts training.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from select_lego_rtc_episodes import select_episodes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "configs/native/pi05-earbuds-50h.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_recipe(path):
    recipe = read_json(path)
    report = ROOT / recipe["report"]
    selection = read_json(report / "selection.json")
    source_ids = read_json(report / "source-episodes.json")
    converted_ids = read_json(report / "episodes.json")
    rows = selection["selected_episodes"]
    if source_ids != [row["source_episode_index"] for row in rows] or len(set(source_ids)) != len(rows):
        raise ValueError("Frozen source IDs differ from selection")
    if converted_ids != list(range(len(rows))):
        raise ValueError("Converted IDs must follow the converter's new contiguous namespace")
    if selection["source_root"] != recipe["source_root"]:
        raise ValueError("Selection source differs from recipe")
    for split in ("train", "val"):
        if sha256(report / f"source-{split}.jsonl") != selection["source_manifest_sha256"][split]:
            raise ValueError(f"Frozen {split} manifest changed")
    source_rows = [
        json.loads(line) for line in (report / "source-train.jsonl").read_text().splitlines() if line.strip()
    ]
    expected = select_episodes(
        [dict(row, episode_index=row["source_episode_index"]) for row in source_rows],
        fps=30,
        hours=50,
        seed=42,
    )
    if rows != [{key: value for key, value in row.items() if key != "episode_index"} for row in expected]:
        raise ValueError("Frozen selection differs from seed42 whole-episode sampling")
    if any(row["task"] != recipe["task"] or row["fps"] != 30 for row in rows):
        raise ValueError("Wrong task or FPS")
    frames = sum(row["length"] for row in rows)
    if frames != selection["frames"] or frames < 50 * 3600 * 30:
        raise ValueError("Selection is shorter than 50h or frame count changed")
    training = recipe["training"]
    if training["config"] != "pi05_yam" or training["horizon"] != 50 or training["fsdp_devices"] != 4:
        raise ValueError("Requires Pi0.5 YAM H50 / four-GPU full finetuning")
    if training["seed"] != selection["seed"] or training["ema_decay"] is not None or training["adam_eps"] != 1e-6:
        raise ValueError("Recipe differs from the native entrypoint's fixed seed/EMA/Adam settings")
    if training["seed"] != 42 or recipe["automatic_training_start"] is not False:
        raise ValueError("Invalid seed or automatic training enabled")
    minimum = (frames * 2 + training["batch_size"] - 1) // training["batch_size"]
    if not minimum <= training["total_steps"] < minimum + 1000:
        raise ValueError("Training budget must cover approximately two frame-start epochs")
    if training["decay_steps"] != training["total_steps"]:
        raise ValueError("Keep the learning-rate schedule fixed across both stops")
    return recipe, report, selection


def check_raw(recipe, selection):
    raw = Path(recipe["source_root"])
    for split in ("train", "val"):
        if sha256(raw / "manifests" / f"{split}.jsonl") != selection["source_manifest_sha256"][split]:
            raise ValueError(f"Server {split} source manifest changed; freeze a new selection")


def require_server(recipe):
    # This workstation must never run preparation imports or training smoke.
    if not Path(recipe["source_root"]).is_dir() or not Path(recipe["python"]).is_file():
        raise RuntimeError("Run this action on the training server using its existing Pi environment")


def check_data(recipe, report, selection):
    check_raw(recipe, selection)
    root = Path(recipe["dataset_root"])
    assets = Path(recipe["assets_root"])
    for filename in ("episodes.json", "source-episodes.json", "selection.json", "contract.json"):
        if sha256(assets / filename) != sha256(report / filename):
            raise ValueError(f"Published selection asset changed: {filename}")
    expected_contract = read_json(report / "contract.json")
    for split in ("train", "val"):
        manifest = read_json(root / split / "conversion_manifest.json")
        if manifest["split"] != split or manifest["source_root"] != recipe["source_root"]:
            raise ValueError("Conversion task or split changed")
        if manifest["contract"] != expected_contract:
            raise ValueError("Conversion physical contract changed")
        if manifest["source_manifest_sha256"] != selection["source_manifest_sha256"][split]:
            raise ValueError("Conversion source changed")
        expected = (
            selection["selected_episodes"]
            if split == "train"
            else [json.loads(line) for line in (report / "source-val.jsonl").read_text().splitlines() if line.strip()]
        )
        actual = manifest["episodes"]
        fields = ("source_episode_index", "source_repo", "source_revision", "length", "task")
        if len(actual) != len(expected) or any(
            row["episode_index"] != i or any(row[key] != source[key] for key in fields)
            for i, (row, source) in enumerate(zip(actual, expected, strict=True))
        ):
            raise ValueError("Conversion episode membership/order changed")
        if manifest["video_encoding"] != (
            "byte-identical independent copies; destination SHA-256 checked; full decode verified"
        ):
            raise ValueError("Requires the existing audited copy-mode conversion")
        info = read_json(root / split / "meta/info.json")
        if info["fps"] != 30 or info["robot_type"] != "yam" or info["codebase_version"] != "v3.0":
            raise ValueError("Wrong published YAM / LeRobot contract")
        for key in ("observation.state", "action"):
            if info["features"][key]["shape"] != [14]:
                raise ValueError("Expected 14D state/action")
        if info["total_frames"] != sum(row["length"] for row in expected) or info["total_episodes"] != len(expected):
            raise ValueError("Published frame/episode counts differ from selection")
    norm_dir = assets / "norm/yam"
    provenance = read_json(norm_dir / "provenance.json")
    if (
        provenance["dataset"] != str(root / "train")
        or provenance["selected_episode_ids"] != read_json(assets / "episodes.json")
        or provenance["selection_sha256"] != sha256(assets / "episodes.json")
        or provenance["validation_split_included"] is not False
        or provenance["frames"] != selection["frames"]
        or provenance["horizon"] != 50
        or provenance["contract"] != expected_contract
        or provenance["delta_mask"] != [True] * 6 + [False] + [True] * 6 + [False]
    ):
        raise ValueError("Norm does not describe this exact train subset / H50 delta contract")
    for filename, field in (("conversion_manifest.json", "manifest_sha256"), ("meta/info.json", "info_sha256")):
        if provenance[field] != sha256(root / "train" / filename):
            raise ValueError("Norm source metadata changed")
    for relative, expected in provenance["parquet_sha256"].items():
        if sha256(root / "train" / relative) != expected:
            raise ValueError(f"Norm source Parquet changed: {relative}")
    checks = read_json(norm_dir / "loader_equivalence.json")
    if not checks or any(row["state_actions_equal"] is not True or row["task"] != recipe["task"] for row in checks):
        raise ValueError("Real loader equivalence checks missing or failed")
    norm = read_json(norm_dir / "norm_stats.json")
    if set(norm["norm_stats"]) != {"state", "actions"}:
        raise ValueError("Invalid norm asset")
    for stats in norm["norm_stats"].values():
        for key in ("mean", "std", "q01", "q99"):
            if len(stats[key]) != 14 or any(not math.isfinite(value) for value in stats[key]):
                raise ValueError("Norm must contain finite 14D statistics")
        if any(low > high for low, high in zip(stats["q01"], stats["q99"], strict=True)):
            raise ValueError("Inverted norm quantiles")
    return {"status": "DATA_READY", "frames": selection["frames"], "norm_sha256": sha256(norm_dir / "norm_stats.json")}


def prepare_data(recipe, report, selection, *, resume=False):
    require_server(recipe)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full conversion/video audit requires a compute-node Slurm job")
    check_raw(recipe, selection)
    assets = Path(recipe["assets_root"])
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for filename in ("episodes.json", "source-episodes.json", "selection.json", "contract.json"):
            target = assets / filename
            if target.exists():
                if sha256(target) != sha256(report / filename):
                    raise ValueError(f"Existing asset differs: {target}")
            else:
                with target.open("xb") as handle:
                    handle.write((report / filename).read_bytes())
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            JAX_PLATFORMS="cpu",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            PYTHONNOUSERSITE="1",
            PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}:{ROOT / 'packages/openpi-client/src'}",
        )
        for split in ("val", "train"):
            command = [
                recipe["python"],
                str(ROOT / "scripts/convert_yam_subset.py"),
                recipe["source_root"],
                str(Path(recipe["dataset_root"]) / split),
                "--split",
                split,
                "--contract",
                str(assets / "contract.json"),
                "--video-mode",
                "copy",
            ]
            if split == "train":
                command += ["--episode-ids", *map(str, read_json(assets / "source-episodes.json"))]
            if resume:
                command += ["--resume"]
            subprocess.run(command, env=environment, cwd=ROOT, check=True)
        norm_dir = assets / "norm/yam"
        if not (norm_dir / "norm_stats.json").exists():
            if norm_dir.exists():
                raise FileExistsError("Incomplete norm directory retained; inspect it before choosing a new output")
            subprocess.run(
                [
                    recipe["python"],
                    str(ROOT / "scripts/compute_yam_norm_stats.py"),
                    "--dataset",
                    str(Path(recipe["dataset_root"]) / "train"),
                    "--train-episodes-file",
                    str(assets / "episodes.json"),
                    "--output",
                    str(norm_dir),
                ],
                env=environment,
                cwd=ROOT,
                check=True,
            )
        print(json.dumps(check_data(recipe, report, selection)))


def train_command(recipe, stop_step, *, resume=False):
    training = recipe["training"]
    if not 0 < stop_step <= training["total_steps"]:
        raise ValueError("Cumulative stop step must be 1..338000")
    assets = Path(recipe["assets_root"])
    command = [
        recipe["python"],
        str(ROOT / "scripts/train_lego_full.py"),
        "--run-name",
        recipe["run_name"],
        "--steps",
        str(stop_step),
        "--dataset",
        str(Path(recipe["dataset_root"]) / "train"),
        "--init-params",
        recipe["init_params"],
        "--assets-dir",
        str(assets / "norm"),
        "--train-episodes-file",
        str(assets / "episodes.json"),
        "--rtc-training-max-delay",
        str(training["rtc_training_max_delay"]),
        "--warmup-steps",
        str(training["warmup_steps"]),
        "--decay-steps",
        str(training["decay_steps"]),
    ]
    if resume:
        command += ["--resume"]
    environment = dict(os.environ)
    for key in ("batch_size", "num_workers", "peak_lr", "decay_lr", "save_interval", "keep_period"):
        environment[f"LEGO_{key.upper()}"] = str(training[key])
    environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts'}:{ROOT / 'packages/openpi-client/src'}"
    return command, environment


def launch(recipe, report, selection, stop_step, *, resume=False):
    require_server(recipe)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires an explicitly submitted four-GPU Slurm job")
    check_data(recipe, report, selection)
    control = Path("/home/wuyan/lyj/YAM/training-runs/control") / recipe["run_name"]
    control.mkdir(parents=True, exist_ok=True)
    with (control / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command, environment = train_command(recipe, stop_step, resume=resume)
        fingerprint = {
            "recipe": recipe,
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "entry_sha256": sha256(ROOT / "scripts/train_lego_full.py"),
            "selection_sha256": sha256(report / "selection.json"),
        }
        fingerprint_path = control / "launch_fingerprint.json"
        if resume:
            if read_json(fingerprint_path) != fingerprint:
                raise ValueError("Code/recipe/selection changed; cannot resume this run")
        else:
            with fingerprint_path.open("x") as handle:
                json.dump(fingerprint, handle, indent=2)
                handle.write("\n")
        with (control / "train.log").open("a") as log:
            subprocess.run(command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    subparsers = parser.add_subparsers(dest="action", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--recipe-only", action="store_true", help="No server/data imports or writes")
    prepare = subparsers.add_parser("prepare-data", help="CPU conversion and norm only; never trains")
    prepare.add_argument("--resume", action="store_true")
    train_parser = subparsers.add_parser("train", help="Explicit Slurm-only training")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--stop-step", type=int, default=None)
    args = parser.parse_args()
    recipe, report, selection = checked_recipe(args.recipe)
    if args.action == "check":
        result = {
            "status": "RECIPE_READY_DATA_UNVERIFIED",
            "episodes": len(selection["selected_episodes"]),
            "hours": selection["hours"],
            "automatic_training_start": False,
        }
        if not args.recipe_only:
            result = check_data(recipe, report, selection)
        print(json.dumps(result))
    elif args.action == "prepare-data":
        prepare_data(recipe, report, selection, resume=args.resume)
    else:
        stop_step = recipe["training"]["first_stop_step"] if args.stop_step is None else args.stop_step
        launch(recipe, report, selection, stop_step, resume=args.resume)


if __name__ == "__main__":
    main()
