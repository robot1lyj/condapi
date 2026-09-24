"""Bind a frozen YAM split and matching norm assets to the existing OpenPI trainer."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/vla-platform/src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vla_platform.project import identifier  # noqa: E402
from vla_platform.project import tree_digest  # noqa: E402
from vla_platform.splits import inspect_split  # noqa: E402


def preflight(recipe_path, split_path, output):
    recipe = json.loads(Path(recipe_path).read_text())
    allowed = {"schema_version", "config", "dataset", "norm_assets", "init_params", "init_params_tree_sha256",
               "num_train_steps", "batch_size", "fsdp_devices", "num_workers"}
    if recipe.get("schema_version") != 1 or set(recipe) != allowed:
        raise ValueError("Invalid modular Pi recipe schema")
    if recipe["config"] != "pi05_yam":
        raise ValueError("Modular Pi full training currently requires pi05_yam")
    for key in ("num_train_steps", "batch_size", "fsdp_devices"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"Invalid {key}")
    if type(recipe["num_workers"]) is not int or recipe["num_workers"] < 0:
        raise ValueError("Invalid num_workers")
    for key in ("dataset", "norm_assets", "init_params"):
        if not isinstance(recipe[key], str) or not Path(recipe[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    dataset = Path(recipe["dataset"]).resolve()
    assets = Path(recipe["norm_assets"]).resolve()
    params = Path(recipe["init_params"]).resolve()
    if not (dataset / "meta/info.json").is_file():
        raise ValueError("Dataset is not a published local LeRobot export")
    if params.name != "params" or not params.is_dir() or not any(params.iterdir()):
        raise ValueError("Expected a nonempty JAX /params directory")
    if not isinstance(recipe["init_params_tree_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", recipe["init_params_tree_sha256"]
    ):
        raise ValueError("Pi base checkpoint tree SHA256 must be explicit")
    norm = assets / "yam/norm_stats.json"
    provenance = assets / "yam/provenance.json"
    if not norm.is_file() or not provenance.is_file():
        raise ValueError("Missing YAM norm stats or provenance")
    split, component, _ = inspect_split(ROOT, split_path)
    if component["source_format"] != "lerobot" or Path(component["source_root"]).resolve() != dataset:
        raise ValueError("Pi recipe dataset differs from the frozen LeRobot split")
    selected = [int(value) for value in split["partitions"]["train"]]
    if not selected or len(selected) != len(set(selected)) or any(str(value) != name for value, name in
                                                         zip(selected, split["partitions"]["train"], strict=True)):
        raise ValueError("Pi train split must contain unique integer LeRobot episode IDs")
    stats = json.loads(provenance.read_text())
    if (stats.get("selected_episode_ids") != sorted(selected)
            or Path(stats.get("dataset", "")).resolve() != dataset
            or stats.get("validation_split_included") is not False
            or stats.get("horizon") != 50
            or stats.get("delta_mask") != [True] * 6 + [False] + [True] * 6 + [False]):
        raise ValueError("YAM norm provenance does not match the train split and action transform")
    output = Path(output).resolve()
    if output.is_relative_to(dataset) or output.is_relative_to(assets) or output.is_relative_to(params):
        raise ValueError("Training output must be outside all source assets")
    if tree_digest(params) != recipe["init_params_tree_sha256"]:
        raise ValueError("Pi base checkpoint tree SHA256 mismatch")
    with norm.open("rb") as stream:
        norm_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return recipe, selected, {"split_manifest": str(Path(split_path).resolve()),
                              "norm_sha256": norm_hash,
                              "norm_provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
                              "init_params_tree_sha256": recipe["init_params_tree_sha256"],
                              "train_episodes": selected}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    identifier(args.run_id)
    recipe, selected, provenance = preflight(args.recipe, args.split_manifest, args.output)
    if args.check_only:
        print(json.dumps({"status": "contract_ready_not_gpu_validated", "provenance": provenance}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Pi training requires an authorized Slurm GPU allocation")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Training artifact directory is not empty")

    import jax  # noqa: PLC0415
    from openpi.training import config as configs  # noqa: PLC0415
    from openpi.training import weight_loaders  # noqa: PLC0415
    import train  # noqa: PLC0415

    if jax.device_count() != recipe["fsdp_devices"] or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("Allocated GPU count does not match the Pi recipe")
    base = configs.get_config(recipe["config"])
    if not isinstance(base.data, configs.LeRobotYamDataConfig):
        raise ValueError("Selected OpenPI config does not use the YAM data adapter")
    data = dataclasses.replace(
        base.data,
        repo_id=recipe["dataset"],
        assets=configs.AssetsConfig(assets_dir=recipe["norm_assets"], asset_id="yam"),
        base_config=dataclasses.replace(base.data.base_config, train_episodes=selected),
    )
    config = dataclasses.replace(
        base,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(recipe["init_params"]),
        exp_name=args.run_id,
        checkpoint_base_dir=str(args.output.resolve() / "checkpoints"),
        batch_size=recipe["batch_size"],
        fsdp_devices=recipe["fsdp_devices"],
        num_workers=recipe["num_workers"],
        num_train_steps=recipe["num_train_steps"],
        wandb_enabled=False,
        overwrite=False,
        resume=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "resolved.json").write_text(json.dumps(dataclasses.asdict(config), default=str, indent=2) + "\n")
    train.main(config)


if __name__ == "__main__":
    main()
