"""Preflight and launch the unchanged XR-1 native trainer on a Slurm GPU node."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.xr1.common import VENDOR  # noqa: E402
from adapters.xr1.common import validate_episode  # noqa: E402
from adapters.xr1.common import validate_stats  # noqa: E402
from adapters.xr1.common import verify_source  # noqa: E402


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_recipe(recipe_path, output):
    """Inspect model/data/kinematics identity without importing or constructing the model."""
    recipe = json.loads(Path(recipe_path).read_text())
    keys = {
        "schema_version", "source_contract", "train_jsons", "stats", "kinematics_audit",
        "pretrained", "checkpoint_sha256", "nproc_per_node", "project", "experiment",
        "max_steps", "batch_size",
    }
    if recipe.get("schema_version") != 1 or set(recipe) - keys:
        raise ValueError("Invalid XR-1 recipe schema")
    if recipe.get("source_contract") != "yam-bimanual-v1":
        raise ValueError("Only explicitly audited YAM training is registered")
    for field in ("nproc_per_node", "max_steps", "batch_size"):
        if type(recipe.get(field)) is not int or recipe[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("project", "experiment"):
        value = recipe.get(field)
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-_"
        if not isinstance(value, str) or not value or any(ch not in alphabet for ch in value):
            raise ValueError(f"Invalid {field}")
    paths = recipe.get("train_jsons")
    if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide a nonempty, unique train JSON list")
    output = Path(output).resolve()
    for field in (*paths, recipe["stats"], recipe["kinematics_audit"], recipe["pretrained"]):
        if not isinstance(field, str) or not Path(field).is_absolute():
            raise ValueError("XR-1 input paths must be absolute")
    inputs = [Path(p).resolve() for p in paths]
    stats_path = Path(recipe["stats"]).resolve()
    audit_path = Path(recipe["kinematics_audit"]).resolve()
    checkpoint = Path(recipe["pretrained"]).resolve()
    for path in (*inputs, stats_path, audit_path, checkpoint):
        if not path.is_file():
            raise ValueError(f"Missing absolute XR-1 input: {path}")
        if output.is_relative_to(path.parent):
            raise ValueError("New training output must be outside the input tree")
    episodes = [validate_episode(path) for path in inputs]
    stats = json.loads(stats_path.read_text())
    validate_stats(stats)
    audit = json.loads(audit_path.read_text())
    required = (
        "source_contract", "source_dataset", "source_revision", "train_episode_ids", "fk_model",
        "fk_model_sha256", "frames_and_units_verified", "target_alignment_verified",
    )
    if any(key not in audit for key in required):
        raise ValueError("Incomplete YAM FK audit")
    if audit["source_contract"] != "yam-bimanual-v1" or any(audit[key] is not True for key in required[-2:]):
        raise ValueError("YAM FK, units and target timing must be verified")
    identity_fields = ("source_dataset", "source_revision", "fk_model_sha256")
    if any(not isinstance(audit[key], str) or not audit[key] for key in identity_fields):
        raise ValueError("YAM source and FK identity must be recorded")
    fk_model = Path(audit["fk_model"])
    if not fk_model.is_absolute() or not fk_model.is_file() or file_hash(fk_model) != audit["fk_model_sha256"]:
        raise ValueError("YAM FK model path/hash mismatch")
    if (not isinstance(audit["train_episode_ids"], list) or len(audit["train_episode_ids"]) != len(inputs)
            or len(set(audit["train_episode_ids"])) != len(inputs)):
        raise ValueError("FK audit must identify every training episode")
    if audit.get("train_json_sha256") != {item["path"]: item["sha256"] for item in episodes}:
        raise ValueError("FK audit does not bind the exact derived JSON files")
    if stats.get("train_json_sha256") != audit["train_json_sha256"]:
        raise ValueError("Normalization statistics do not bind the training split")
    if recipe.get("checkpoint_sha256") != "94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7":
        raise ValueError("XR-1 checkpoint identity differs from the selected official 5B revision")
    return recipe, stats, {"episodes": episodes, "audit_sha256": file_hash(audit_path),
                           "stats_sha256": file_hash(stats_path), "upstream_revision": verify_source()}


def compose_config(recipe, stats, output):
    """Use XR-1's Hydra configuration, replacing demo data and all demo statistics."""
    from hydra import compose  # noqa: PLC0415
    from hydra import initialize_config_dir  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    with initialize_config_dir(version_base=None, config_dir=str(VENDOR / "configs")):
        config = compose(config_name="config")
    data = config.data.params.train_datasets
    data.paths = recipe["train_jsons"]
    data.batch_size = recipe["batch_size"]
    for key in ("mean", "std", "q01", "q99"):
        data[key] = stats[key]
    config.model.params.pretrained = recipe["pretrained"]
    config.trainer.max_steps = recipe["max_steps"]
    config.trainer.project = recipe["project"]
    config.trainer.exp_name = recipe["experiment"]
    config.trainer.default_root_dir = str(output / "native")
    return OmegaConf.to_container(config, resolve=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    recipe, stats, provenance = inspect_recipe(args.recipe, args.output)
    if file_hash(recipe["pretrained"]) != recipe["checkpoint_sha256"]:
        raise ValueError("XR-1 pretrained checkpoint SHA256 mismatch")
    config = compose_config(recipe, stats, args.output.resolve())
    if args.check_only:
        print(json.dumps({"status": "config_ready_not_gpu_validated", "provenance": provenance}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("XR-1 training requires an authorized Slurm GPU allocation")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Training artifact directory is not empty")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "assets").mkdir()
    from omegaconf import OmegaConf  # noqa: PLC0415

    OmegaConf.save(OmegaConf.create(config), args.output / "resolved.yaml")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(VENDOR) + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["MLP_WORKER_NUM"] = "1"
    env["MLP_WORKER_GPU"] = str(recipe["nproc_per_node"])
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         f"--nproc_per_node={recipe['nproc_per_node']}", str(VENDOR / "tools/train.py"),
         "--config-path", str(args.output.resolve()), "--config-name", "resolved"],
        cwd=args.output, env=env, check=True,
    )


if __name__ == "__main__":
    main()
