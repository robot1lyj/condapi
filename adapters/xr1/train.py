"""Preflight and launch the unchanged XR-1 native trainer on a Slurm GPU node."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.xr1.common import VENDOR
from adapters.xr1.common import validate_episode
from adapters.xr1.common import validate_stats
from adapters.xr1.common import verify_source


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_recipe(recipe_path, output):
    """Inspect model/data/kinematics identity without importing or constructing the model."""
    recipe = json.loads(Path(recipe_path).read_text())
    keys = {
        "schema_version",
        "source_contract",
        "train_jsons",
        "train_manifest",
        "stats",
        "kinematics_audit",
        "pretrained",
        "checkpoint_sha256",
        "nproc_per_node",
        "project",
        "experiment",
        "max_steps",
        "batch_size",
        "gradient_accumulation",
        "save_interval",
        "selection_sha256",
        "source_manifest_sha256",
        "expected_episodes",
        "expected_frames",
    }
    if recipe.get("schema_version") != 1 or set(recipe) - keys:
        raise ValueError("Invalid XR-1 recipe schema")
    if recipe.get("source_contract") != "yam-bimanual-v1":
        raise ValueError("Only explicitly audited YAM training is registered")
    for field in ("nproc_per_node", "max_steps", "batch_size"):
        if type(recipe.get(field)) is not int or recipe[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if type(recipe.get("gradient_accumulation", 1)) is not int or recipe.get("gradient_accumulation", 1) < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    if "save_interval" in recipe and (type(recipe["save_interval"]) is not int or recipe["save_interval"] < 1):
        raise ValueError("save_interval must be a positive integer")
    for field in ("project", "experiment"):
        value = recipe.get(field)
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-_"
        if not isinstance(value, str) or not value or any(ch not in alphabet for ch in value):
            raise ValueError(f"Invalid {field}")
    if ("train_jsons" in recipe) == ("train_manifest" in recipe):
        raise ValueError("Provide exactly one of train_jsons or train_manifest")
    manifest = None
    if "train_manifest" in recipe:
        manifest_path = Path(recipe["train_manifest"])
        if not manifest_path.is_absolute() or not manifest_path.is_file():
            raise ValueError("Missing absolute XR-1 train manifest")
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("episodes")
        ids = manifest.get("episode_ids")
        if (
            manifest.get("schema") != "yam_xr1_lego_eef_v1"
            or manifest.get("split") != "train"
            or not isinstance(entries, list)
            or not entries
            or not isinstance(ids, list)
            or len(entries) != len(ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Incomplete or invalid 50h train manifest")
        for key in ("selection_sha256", "source_manifest_sha256"):
            if manifest.get(key) != recipe.get(key):
                raise ValueError(f"50h train manifest {key} differs from recipe")
        if (
            len(entries) != recipe.get("expected_episodes")
            or sum(entry["frames"] for entry in entries) != recipe.get("expected_frames")
            or [entry["episode_index"] for entry in entries] != ids
        ):
            raise ValueError("50h train episode identity or frame total differs from recipe")
        paths = [entry["json"] for entry in entries]
    else:
        paths = recipe["train_jsons"]
    if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide a nonempty, unique train JSON list")
    output = Path(output).resolve()
    input_fields = [*paths, recipe["stats"], recipe["kinematics_audit"], recipe["pretrained"]]
    if manifest is not None:
        input_fields.append(recipe["train_manifest"])
    for field in input_fields:
        if not isinstance(field, str) or not Path(field).is_absolute():
            raise ValueError("XR-1 input paths must be absolute")
    inputs = [Path(p).resolve() for p in paths]
    stats_path = Path(recipe["stats"]).resolve()
    audit_path = Path(recipe["kinematics_audit"]).resolve()
    for path in (Path(p).resolve() for p in input_fields):
        if not path.is_file():
            raise ValueError(f"Missing absolute XR-1 input: {path}")
        if output.is_relative_to(path.parent):
            raise ValueError("New training output must be outside the input tree")
    episodes = [validate_episode(path) for path in inputs]
    if manifest is not None:
        for entry, episode in zip(manifest["episodes"], episodes, strict=True):
            if (
                entry["json"] != episode["path"]
                or entry["json_sha256"] != episode["sha256"]
                or entry["frames"] != episode["frames"]
            ):
                raise ValueError("50h train manifest does not bind the derived episode")
    stats = json.loads(stats_path.read_text())
    validate_stats(stats)
    audit = json.loads(audit_path.read_text())
    required = (
        "source_contract",
        "source_dataset",
        "source_revision",
        "train_episode_ids",
        "fk_model",
        "fk_model_sha256",
        "frames_and_units_verified",
        "target_alignment_verified",
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
    if (
        not isinstance(audit["train_episode_ids"], list)
        or len(audit["train_episode_ids"]) != len(inputs)
        or len(set(audit["train_episode_ids"])) != len(inputs)
    ):
        raise ValueError("FK audit must identify every training episode")
    if audit.get("train_json_sha256") != {item["path"]: item["sha256"] for item in episodes}:
        raise ValueError("FK audit does not bind the exact derived JSON files")
    if manifest is not None and (
        audit["train_episode_ids"] != manifest["episode_ids"]
        or audit["source_dataset"] != manifest["source_repo"]
        or audit["source_revision"] != manifest["source_manifest_sha256"]
        or audit["fk_model"] != manifest["fk_model"]
        or audit["fk_model_sha256"] != manifest["fk_model_sha256"]
        or stats.get("train_manifest_sha256") != file_hash(recipe["train_manifest"])
    ):
        raise ValueError("50h FK audit or statistics differ from the train manifest")
    if stats.get("train_json_sha256") != audit["train_json_sha256"]:
        raise ValueError("Normalization statistics do not bind the training split")
    if recipe.get("checkpoint_sha256") != "94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7":
        raise ValueError("XR-1 checkpoint identity differs from the selected official 5B revision")
    resolved = {**recipe, "train_jsons": paths}
    return (
        resolved,
        stats,
        {
            "episodes": episodes,
            "audit_sha256": file_hash(audit_path),
            "stats_sha256": file_hash(stats_path),
            "upstream_revision": verify_source(),
            "train_manifest_sha256": file_hash(recipe["train_manifest"]) if manifest else None,
        },
    )


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
    accumulation = recipe.get("gradient_accumulation", 1)
    config.trainer.accumulate_grad_batches = accumulation
    # The native dataset sizes itself in micro-batches, while Trainer counts
    # optimizer steps. Without this adjustment, accumulation truncates the data.
    config.data.params.max_steps = recipe["max_steps"] * accumulation
    if "save_interval" in recipe:
        config.trainer.save_interval = recipe["save_interval"]
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
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={recipe['nproc_per_node']}",
            str(VENDOR / "tools/train.py"),
            "--config-path",
            str(args.output.resolve()),
            "--config-name",
            "resolved",
        ],
        cwd=args.output,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
