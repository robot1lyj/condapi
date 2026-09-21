"""Pinned Lego full-finetuning entrypoint. Launch only inside a four-GPU allocation."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re

import jax
from select_lego_rtc_episodes import read_episode_ids
import train

from openpi.training import checkpoints
from openpi.training import config as configs
from openpi.training import optimizer
from openpi.training import weight_loaders


def make_config(
    *,
    resume=False,
    steps=40_000,
    run_name="lego_full_b32",
    init_params=None,
    train_episodes=None,
    rtc_training_max_delay=0,
    warmup_steps=None,
    decay_steps=None,
    assets_dir=None,
):
    config = configs.get_config("pi05_yam")
    batch_size = int(os.environ.get("LEGO_BATCH_SIZE", "32"))
    num_workers = int(os.environ.get("LEGO_NUM_WORKERS", "2"))
    save_interval = int(os.environ.get("LEGO_SAVE_INTERVAL", "2000"))
    keep_period = int(os.environ.get("LEGO_KEEP_PERIOD", "20000"))
    peak_lr = float(os.environ.get("LEGO_PEAK_LR", "1.25e-5"))
    decay_lr = float(os.environ.get("LEGO_DECAY_LR", "1.25e-6"))
    if batch_size < 32:
        raise ValueError("LEGO_BATCH_SIZE must be at least 32")
    if num_workers < 0:
        raise ValueError("LEGO_NUM_WORKERS must be non-negative")
    if save_interval <= 0 or keep_period <= 0:
        raise ValueError("LEGO_SAVE_INTERVAL and LEGO_KEEP_PERIOD must be positive")
    if peak_lr <= 0 or decay_lr <= 0:
        raise ValueError("LEGO_PEAK_LR and LEGO_DECAY_LR must be positive")
    if rtc_training_max_delay and init_params is None:
        raise ValueError("RTC adaptation requires explicit --init-params; no implicit base restart")
    adaptation = init_params is not None
    if adaptation:
        params = Path(init_params).expanduser().resolve()
        if params.name != "params" or not params.is_dir():
            raise ValueError("--init-params must point to an existing JAX checkpoint params directory")
        explicit_assets = assets_dir is not None
        assets_dir = str(Path(assets_dir).expanduser().resolve()) if explicit_assets else str(params.parent / "assets")
        if not (Path(assets_dir) / "yam" / "norm_stats.json").is_file():
            raise ValueError("Missing assets/yam/norm_stats.json: supply --assets-dir or parent checkpoint assets")
        if explicit_assets and train_episodes is not None:
            provenance_path = Path(assets_dir) / "yam" / "provenance.json"
            if not provenance_path.is_file():
                raise ValueError("Explicit subset assets require compute_yam_norm_stats.py provenance.json")
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("selected_episode_ids") != sorted(train_episodes):
                raise ValueError("Norm episode selection differs from training subset")
        init_params = str(params)
        peak_lr = float(os.environ.get("LEGO_PEAK_LR", "3e-6"))
        decay_lr = float(os.environ.get("LEGO_DECAY_LR", "3e-7"))
    else:
        if assets_dir is not None:
            raise ValueError("--assets-dir requires explicit --init-params")
        assets_dir = "/home/wuyan/lyj/YAM/training-assets/lego_lerobot_v1_20260907/pi05_h50"
    warmup_steps = (200 if adaptation else 1000) if warmup_steps is None else warmup_steps
    decay_steps = (steps if adaptation else 162_097) if decay_steps is None else decay_steps
    if not 0 <= warmup_steps < decay_steps or min(peak_lr, decay_lr) <= 0:
        raise ValueError("Require 0 <= warmup_steps < decay_steps and positive learning rates")
    return dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, rtc_training_max_delay=rtc_training_max_delay),
        exp_name=run_name,
        checkpoint_base_dir="/home/wuyan/lyj/YAM/training-runs",
        batch_size=batch_size,
        fsdp_devices=4,
        num_workers=num_workers,
        ema_decay=None,
        num_train_steps=steps,
        save_interval=save_interval,
        keep_period=keep_period,
        log_interval=10,
        seed=42,
        wandb_enabled=False,
        resume=resume,
        overwrite=False,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps, peak_lr=peak_lr, decay_steps=decay_steps, decay_lr=decay_lr
        ),
        optimizer=dataclasses.replace(config.optimizer, eps=1e-6),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            init_params or "/home/wuyan/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
        ),
        data=dataclasses.replace(
            config.data,
            repo_id="/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train",
            assets=configs.AssetsConfig(assets_dir=assets_dir, asset_id="yam"),
            base_config=dataclasses.replace(config.data.base_config, train_episodes=train_episodes),
        ),
    )


def check_training_contract(config, control):
    """Pin the objective, subset, parent and norm across restarts of a new run."""
    norm_path = Path(config.data.assets.assets_dir) / "yam" / "norm_stats.json"
    record = {
        "model": dataclasses.asdict(config.model),
        "parent_params": config.weight_loader.params_path,
        "repo_id": config.data.repo_id,
        "train_episodes": config.data.base_config.train_episodes,
        "norm_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "optimizer": dataclasses.asdict(config.optimizer),
        "batch_size": config.batch_size,
        "seed": config.seed,
    }
    path = control / "training_contract.json"
    if config.resume:
        if not path.exists():
            if config.model.rtc_training_max_delay:
                raise RuntimeError("RTC resume requires the original training_contract.json")
            return  # Existing legacy runs predate this contract.
        if json.loads(path.read_text()) != record:
            raise RuntimeError("Training contract changed: use a new run instead of --resume")
    else:
        with path.open("x") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")


def require_resume_checkpoint(config):
    if not config.checkpoint_dir.exists():
        raise RuntimeError("Resume requested but run directory is missing")
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True
    )
    manager.close()
    if not resuming:
        raise RuntimeError("No committed checkpoint: refusing to restart from base under --resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default=os.environ.get("LEGO_RUN_NAME", "lego_full_b32"))
    parser.add_argument("--steps", type=int, default=40_000, help="Cumulative stop step, not additional steps")
    parser.add_argument("--init-params", help="Parent JAX checkpoint /params; creates a new optimizer/run")
    parser.add_argument("--train-episodes-file", type=Path, help="JSON list of complete source train episode IDs")
    parser.add_argument(
        "--rtc-training-max-delay", type=int, default=0, help="Inclusive bound; 0 disables RTC training"
    )
    parser.add_argument("--assets-dir", help="Explicit norm assets root (contains yam/norm_stats.json)")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--decay-steps", type=int, help="Keep fixed when extending/resuming the same run")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_name):
        parser.error("invalid run name")
    if args.steps <= 0:
        parser.error("steps must be positive")
    if jax.device_count() != 4 or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("This entrypoint requires four allocated GPUs")
    episodes = read_episode_ids(args.train_episodes_file) if args.train_episodes_file else None
    config = make_config(
        resume=args.resume,
        steps=args.steps,
        run_name=args.run_name,
        init_params=args.init_params,
        train_episodes=episodes,
        rtc_training_max_delay=args.rtc_training_max_delay,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        assets_dir=args.assets_dir,
    )
    if args.resume:
        require_resume_checkpoint(config)
    control = Path("/home/wuyan/lyj/YAM/training-runs/control") / args.run_name
    control.mkdir(parents=True, exist_ok=True)
    manifest = control / ("resume_config.json" if args.resume else "initial_config.json")
    if not args.resume and manifest.exists():
        raise RuntimeError("Existing launch manifest: inspect the prior attempt before restarting")
    check_training_contract(config, control)
    manifest.write_text(json.dumps(dataclasses.asdict(config), default=str, indent=2) + "\n")
    print(f"RUN_DIRECTORY={config.checkpoint_dir}", flush=True)
    train.main(config)


if __name__ == "__main__":
    main()
