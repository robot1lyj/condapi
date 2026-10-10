"""Preflight and launch XR-1's native trainer on a Slurm GPU node."""

import argparse
import fcntl
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
        "keep_period",
        "selection_sha256",
        "source_manifest_sha256",
        "expected_episodes",
        "expected_frames",
        "rtc_mode",
        "async_prefix_min",
        "async_prefix_max",
        "optimizer_offload",
        "communication_bucket_size",
        "logger_backend",
        "fk_verification_scope",
    }
    if recipe.get("schema_version") != 1 or set(recipe) - keys:
        raise ValueError("Invalid XR-1 recipe schema")
    if recipe.get("source_contract") != "yam-bimanual-v1":
        raise ValueError("Only explicitly audited YAM training is registered")
    if recipe.get("rtc_mode", "native_async") not in ("native_async", "disabled"):
        raise ValueError("rtc_mode must be native_async or disabled")
    prefix_min, prefix_max = recipe.get("async_prefix_min", 1), recipe.get("async_prefix_max", 6)
    if type(prefix_min) is not int or type(prefix_max) is not int or not 1 <= prefix_min <= prefix_max < 30:
        raise ValueError("async prefix must satisfy 1 <= min <= max < 30")
    for field in ("nproc_per_node", "max_steps", "batch_size"):
        if type(recipe.get(field)) is not int or recipe[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if type(recipe.get("gradient_accumulation", 1)) is not int or recipe.get("gradient_accumulation", 1) < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    if "save_interval" in recipe and (type(recipe["save_interval"]) is not int or recipe["save_interval"] < 1):
        raise ValueError("save_interval must be a positive integer")
    if "keep_period" in recipe and (
        type(recipe["keep_period"]) is not int
        or recipe["keep_period"] < 1
        or recipe["keep_period"] % recipe.get("save_interval", 10000)
    ):
        raise ValueError("keep_period must be a positive multiple of save_interval")
    if type(recipe.get("optimizer_offload", False)) is not bool:
        raise ValueError("optimizer_offload must be boolean")
    if recipe.get("logger_backend", "wandb") not in ("csv", "wandb"):
        raise ValueError("logger_backend must be csv or wandb")
    if recipe.get("fk_verification_scope", "independent") not in ("independent", "documented_nominal_training"):
        raise ValueError("Unrecognized FK verification scope")
    if type(recipe.get("communication_bucket_size", 500000000)) is not int or recipe.get("communication_bucket_size", 500000000) < 1:
        raise ValueError("communication_bucket_size must be a positive integer")
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
    if audit["source_contract"] != "yam-bimanual-v1":
        raise ValueError("Wrong FK source contract")
    if recipe.get("fk_verification_scope", "independent") == "independent":
        if any(audit[key] is not True for key in required[-2:]):
            raise ValueError("Independent YAM units and target timing must be verified")
    # A documented nominal experiment is explicit in the recipe. It must
    # never turn missing raw-unit/capture-time evidence into verified facts.
    elif (audit.get("verification_scope") != "documented_nominal_training"
            or any(audit.get(key) is not True for key in (
                "nominal_fk_verified", "source_row_alignment_verified", "documented_unit_contract_verified"))
            or audit["frames_and_units_verified"] is not False
            or audit["target_alignment_verified"] is not False
            or audit.get("robot_calibration_verified") is not False
            or audit.get("raw_command_capture_timing_verified") is not False):
        raise ValueError("Incomplete nominal FK audit or falsely promoted independent verification")
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
            "upstream_manifest_sha256": file_hash(VENDOR / "UPSTREAM.json"),
            "train_manifest_sha256": file_hash(recipe["train_manifest"]) if manifest else None,
        },
    )


def compose_config(recipe, stats, output):
    """Use XR-1's Hydra configuration, replacing demo data and all demo statistics."""
    from hydra import compose  # noqa: PLC0415
    from hydra import initialize_config_dir  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from omegaconf import open_dict  # noqa: PLC0415

    with initialize_config_dir(version_base=None, config_dir=str(VENDOR / "configs")):
        config = compose(config_name="config")
    data = config.data.params.train_datasets
    data.paths = recipe["train_jsons"]
    data.batch_size = recipe["batch_size"]
    for key in ("mean", "std", "q01", "q99"):
        data[key] = stats[key]
    config.model.params.pretrained = recipe["pretrained"]
    with open_dict(config.model.params.model):
        config.model.params.model.async_train = recipe.get("rtc_mode", "native_async") == "native_async"
        config.model.params.model.async_prefix_min = recipe.get("async_prefix_min", 1)
        config.model.params.model.async_prefix_max = recipe.get("async_prefix_max", 6)
    config.trainer.max_steps = recipe["max_steps"]
    accumulation = recipe.get("gradient_accumulation", 1)
    config.trainer.accumulate_grad_batches = accumulation
    if recipe.get("optimizer_offload", False):
        with open_dict(config.trainer.strategy.params):
            config.trainer.strategy.params.stage = 2
            config.trainer.strategy.params.offload_optimizer = True
            config.trainer.strategy.params.pin_memory = True
        config.trainer.optimizer.type = "deepspeed.ops.adam.DeepSpeedCPUAdam"
    bucket = recipe.get("communication_bucket_size", 500000000)
    config.trainer.strategy.params.allgather_bucket_size = bucket
    config.trainer.strategy.params.reduce_bucket_size = bucket
    # The native dataset sizes itself in micro-batches, while Trainer counts
    # optimizer steps. Without this adjustment, accumulation truncates the data.
    config.data.params.max_steps = recipe["max_steps"] * accumulation
    if "save_interval" in recipe:
        config.trainer.save_interval = recipe["save_interval"]
    if "keep_period" in recipe:
        with open_dict(config.trainer):
            config.trainer.keep_period = recipe["keep_period"]
    with open_dict(config.trainer):
        config.trainer.logger_backend = recipe.get("logger_backend", "wandb")
        config.trainer.log_every_n_steps = 1
    config.trainer.project = recipe["project"]
    config.trainer.exp_name = recipe["experiment"]
    config.trainer.default_root_dir = str(output / "native")
    return OmegaConf.to_container(config, resolve=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Restore the same run's committed last.ckpt full state")
    parser.add_argument("--stop-after", type=int, help="Stop at this cumulative step without changing the run's LR/data contract")
    args = parser.parse_args(argv)
    recipe, stats, provenance = inspect_recipe(args.recipe, args.output)
    if file_hash(recipe["pretrained"]) != recipe["checkpoint_sha256"]:
        raise ValueError("XR-1 pretrained checkpoint SHA256 mismatch")
    config = compose_config(recipe, stats, args.output.resolve())
    if args.stop_after is not None and not 0 < args.stop_after <= recipe["max_steps"]:
        parser.error("stop-after must be between 1 and recipe max_steps")
    output = args.output.resolve()
    if args.resume:
        sys.path.insert(0, str(VENDOR))
        from mibot.utils.checkpoint_contract import validate_checkpoint  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        if "keep_period" not in recipe:
            parser.error("Resume requires committed periodic checkpoints")
        saved_config = OmegaConf.to_container(OmegaConf.load(output / "resolved.yaml"), resolve=True)
        if saved_config != config or json.loads((output / "provenance.json").read_text()) != provenance:
            parser.error("Resume requires the original recipe, dataset, statistics, audit and source")
        native = output / "native" / f"project_{recipe['project']}" / recipe["experiment"]
        last = native / "last.ckpt"
        if not last.is_symlink() or last.resolve().parent != native:
            parser.error("Missing or foreign last.ckpt recovery link")
        receipt = validate_checkpoint(last, recipe["nproc_per_node"])
        if receipt["global_step"] >= recipe["max_steps"]:
            parser.error("Run already reached its target step")
        if args.stop_after is not None and args.stop_after <= receipt["global_step"]:
            parser.error("stop-after must exceed the recovery step")
        config["trainer"]["ckpt_path"] = str(last.resolve())
    if args.check_only:
        print(json.dumps({"status": "config_ready_not_gpu_validated", "provenance": provenance}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("XR-1 training requires an authorized Slurm GPU allocation")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock outside the run: a second launcher must never resume or
    # initialize the same native checkpoint tree while this process is alive.
    lock = (output.parent / f".{output.name}.train.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another launcher owns this run")
    if args.resume:
        if str(last.resolve()) != config["trainer"]["ckpt_path"]:
            parser.error("Recovery point changed during preflight; rerun --resume")
        validate_checkpoint(last, recipe["nproc_per_node"])
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error("Training artifact directory is not empty; use --resume for the same run")
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=args.resume)
    from omegaconf import OmegaConf  # noqa: PLC0415

    config_name = "resolved-resume" if args.resume else "resolved"
    OmegaConf.save(OmegaConf.create(config), output / f"{config_name}.yaml")
    if not args.resume:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.stop_after is not None:
        # Keep the canonical scheduler and data length intact across segments.
        config["trainer"]["max_steps"] = args.stop_after
        config_name += "-stop"
        OmegaConf.save(OmegaConf.create(config), output / f"{config_name}.yaml")
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
            config_name,
        ],
        cwd=args.output,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
