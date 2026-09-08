"""Delegate to native LeRobot training; no model, dataset or optimizer reimplementation."""

import argparse
import json
from pathlib import Path
import runpy
import sys


def native_arguments(config, policy_type, output):
    """Keep the complete upstream config, overriding only project-owned side effects."""
    config = Path(config).resolve()
    value = json.loads(config.read_text(encoding="utf-8"))
    if value.get("policy", {}).get("type") != policy_type:
        raise ValueError("Native policy.type does not match the selected model")
    if value.get("resume", False):
        raise ValueError("This launcher starts new runs; resume requires a separate native workflow")
    if value.get("reward_model") or (value.get("job") or {}).get("target", "local") != "local":
        raise ValueError("Only local policy training is supported by this launcher")
    return [
        f"--config_path={config}",
        f"--output_dir={Path(output).resolve()}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
        "--save_checkpoint_to_hub=false",
        "--job.target=local",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-type", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    native = native_arguments(args.config, args.policy_type, args.output)
    previous = sys.argv
    try:
        sys.argv = ["lerobot-train", *native]
        # All heavy imports and native processor/checkpoint behavior stay in this child.
        runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
