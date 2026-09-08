"""Pinned Lego full-finetuning entrypoint. Launch only inside a four-GPU allocation."""

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import train

from openpi.training import config as configs
from openpi.training import optimizer
from openpi.training import weight_loaders


def make_config(*, resume=False, steps=40_000):
    config = configs.get_config("pi05_yam")
    return dataclasses.replace(
        config,
        exp_name="lego_full_b64",
        checkpoint_base_dir="/home/wuyan/lyj/YAM/training-runs",
        batch_size=64,
        fsdp_devices=4,
        num_workers=8,
        ema_decay=None,
        num_train_steps=steps,
        save_interval=20_000,
        keep_period=20_000,
        log_interval=10,
        seed=42,
        wandb_enabled=False,
        resume=resume,
        overwrite=False,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=2.5e-5, decay_steps=162_097, decay_lr=2.5e-6
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/wuyan/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
        ),
        data=dataclasses.replace(
            config.data,
            repo_id="/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train",
            assets=configs.AssetsConfig(
                assets_dir="/home/wuyan/lyj/YAM/training-assets/lego_lerobot_v1_20260907/pi05_h50", asset_id="yam"
            ),
            base_config=dataclasses.replace(config.data.base_config, train_episodes=None),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=40_000, help="Cumulative stop step, not additional steps")
    args = parser.parse_args()
    if not 0 < args.steps <= 162_097:
        parser.error("steps must be within 1..162097")
    if jax.device_count() != 4 or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("This entrypoint requires four allocated GPUs")
    config = make_config(resume=args.resume, steps=args.steps)
    if args.resume and not config.checkpoint_dir.exists():
        raise RuntimeError("Resume requested but run directory is missing")
    control = Path("/home/wuyan/lyj/YAM/training-runs/control/lego_full_b64")
    control.mkdir(parents=True, exist_ok=True)
    manifest = control / ("resume_config.json" if args.resume else "initial_config.json")
    if not args.resume and manifest.exists():
        raise RuntimeError("Existing launch manifest: inspect the prior attempt before restarting")
    manifest.write_text(json.dumps(dataclasses.asdict(config), default=str, indent=2) + "\n")
    print(f"RUN_DIRECTORY={config.checkpoint_dir}", flush=True)
    train.main(config)


if __name__ == "__main__":
    main()
