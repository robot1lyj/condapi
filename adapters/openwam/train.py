"""Plan/launch native OpenWAM fine-tuning in a dedicated server Conda environment."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.openwam.common import VENDOR
from adapters.openwam.common import activate_source
from adapters.openwam.common import read_info
from adapters.openwam.common import validate_data_config
from adapters.openwam.common import verify_source


def compose_config(recipe_path, output):
    """Compose native defaults, inherit checkpoint architecture, then apply explicit overrides."""
    from hydra import compose  # noqa: PLC0415
    from hydra import initialize_config_dir  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    recipe = json.loads(Path(recipe_path).read_text())
    allowed = {"schema_version", "hydra_overrides", "model", "dataloader", "training", "project", "nproc_per_node"}
    if recipe.get("schema_version") != 1 or set(recipe) - allowed:
        raise ValueError("Invalid OpenWAM recipe schema")
    workers = recipe.get("nproc_per_node")
    if type(workers) is not int or workers < 1:
        raise ValueError("nproc_per_node must be an explicit positive integer")
    training = recipe.get("training", {})
    finetune, resume = training.get("finetune_ckpt_path"), training.get("resume_ckpt_path")
    if finetune and resume:
        raise ValueError("finetune and resume are mutually exclusive")
    source = finetune or resume
    with initialize_config_dir(version_base=None, config_dir=str(VENDOR / "configs")):
        cfg = compose(config_name="train", overrides=recipe.get("hydra_overrides", []))
    saved = None
    if source:
        source = Path(source)
        if not source.is_absolute() or not (source / "config.yaml").is_file():
            raise ValueError("Checkpoint must be an absolute self-contained native run directory")
        if not any(source.glob("checkpoint_step_*.safetensors")):
            raise ValueError("Checkpoint contains no native weights")
        saved = OmegaConf.load(source / "config.yaml")
        if recipe.get("hydra_overrides"):
            raise ValueError("Warm-start/resume inherits checkpoint model; use explicit model overrides")
        cfg.model = saved.model
        if resume:
            cfg.training = saved.training
            cfg.project = saved.project
            cfg.dataloader = saved.dataloader
    cfg = OmegaConf.merge(cfg, {k: recipe[k] for k in ("model", "training", "project") if k in recipe})
    if "dataloader" in recipe:
        cfg.dataloader = OmegaConf.create(recipe["dataloader"])
    if Path(output).resolve().is_relative_to(Path(cfg.dataloader.dataset_dir).resolve()):
        raise ValueError("Output must be outside the source dataset")
    if source and Path(output).resolve().is_relative_to(Path(source).resolve()):
        raise ValueError("New launcher artifacts must be outside the source checkpoint")
    info = read_info(cfg.dataloader.dataset_dir)
    cfg.dataloader.fps = info["fps"]
    plain = OmegaConf.to_container(cfg, resolve=True)
    validate_data_config(plain["dataloader"], plain["model"]["architecture"])
    for key in ("dataset_dir", "normalization_json"):
        if not Path(plain["dataloader"][key]).is_absolute():
            raise ValueError(f"dataloader.{key} must be absolute")
    if saved is not None:
        old = OmegaConf.to_container(saved, resolve=True)
        for key in ("action_dim", "state_dim", "framework", "variant"):
            if old["model"]["architecture"].get(key) != plain["model"]["architecture"].get(key):
                raise ValueError(f"Checkpoint {key} mismatch; implicit action-head resizing is forbidden")
        if resume:
            if (
                old.get("condapi", {}).get("upstream_revision")
                != json.loads((VENDOR / "UPSTREAM.json").read_text())["revision"]
            ):
                raise ValueError("Resume requires the original OpenWAM source revision")
            if old.get("condapi", {}).get("nproc_per_node") != workers:
                raise ValueError("Resume changed world size or lacks the original launcher contract")
            # Resume retains the exact model, data and optimizer/scheduler contract.
            ignored = {"finetune_ckpt_path", "resume_ckpt_path", "output_path"}
            for section in ("model", "dataloader", "training", "project"):
                before, after = dict(old[section]), dict(plain[section])
                skip = (
                    ignored
                    if section == "training"
                    else {"normalization_stats_path"}
                    if section == "dataloader"
                    else set()
                )
                if {k: v for k, v in before.items() if k not in skip} != {
                    k: v for k, v in after.items() if k not in skip
                }:
                    raise ValueError(f"Resume changed {section}; use a new finetune run")
            if not any(source.glob("accel_state_step_*")):
                raise ValueError("Resume requires native Accelerate full-state checkpoint")
    OmegaConf.update(
        cfg,
        "condapi",
        {
            "nproc_per_node": workers,
            "adapter": "yam_lerobot_v1",
            "upstream_revision": json.loads((VENDOR / "UPSTREAM.json").read_text())["revision"],
        },
        force_add=True,
    )
    cfg.training.output_path = str(Path(output).resolve() / "checkpoints")
    cfg.training.save_full_states_for_resume = True
    cfg.dataloader.normalization_stats_path = str(Path(output).resolve() / "normalization_stats.npy")
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), workers


def worker(config):
    activate_source()
    from omegaconf import OmegaConf  # noqa: PLC0415
    from openwam.dataloader.registry import register_dataset  # noqa: PLC0415
    from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: PLC0415

    from adapters.openwam.data import YamDataset  # noqa: PLC0415
    from adapters.openwam.metrics import metrics_hook  # noqa: PLC0415

    register_dataset("yam_lerobot")(YamDataset)
    OpenWAMTrainer.log_step = metrics_hook(OpenWAMTrainer.log_step)
    spec = importlib.util.spec_from_file_location("condapi_openwam_native_train", VENDOR / "scripts/train.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    # Invoke the upstream entry (including its exception handling/process-group cleanup).
    native.main.__wrapped__(OmegaConf.load(config))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker and args.check_only:
        parser.error("--worker and --check-only cannot be combined")
    activate_source()
    revision = verify_source()
    if not args.check_only and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Training is restricted to an authorized Slurm compute allocation; local training is forbidden")
    if args.worker:
        worker(args.config)
        return
    if args.output is None:
        parser.error("--output is required")
    cfg, workers = compose_config(args.config, args.output)
    from omegaconf import OmegaConf  # noqa: PLC0415

    if args.check_only:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    import numpy as np  # noqa: PLC0415

    from adapters.openwam.data import YamDataset  # noqa: PLC0415
    from adapters.openwam.data import load_stats  # noqa: PLC0415

    # Metadata/hash preflight, no training and no source writes.
    YamDataset(OmegaConf.to_container(cfg.dataloader, resolve=True))
    payload = load_stats(cfg.dataloader.normalization_json)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "normalization_stats.npy", payload["stats"])
    resolved = args.output.resolve() / "resolved.yaml"
    OmegaConf.save(cfg, resolved)
    (args.output / "source.json").write_text(
        json.dumps(
            {"upstream_revision": revision, "recipe": json.loads(args.config.read_text()), "dataset_audit": payload},
            indent=2,
        )
        + "\n"
    )
    env = dict(os.environ, WANDB_MODE="disabled", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={workers}",
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(resolved),
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
