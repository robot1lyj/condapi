#!/usr/bin/env python3
"""Benchmark OpenArm LeRobot data loading for a training config."""

import argparse
import dataclasses
import time

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _override_video_backend(config: _config.TrainConfig, backend: str | None) -> _config.TrainConfig:
    if backend is None:
        return config
    if not hasattr(config.data, "base_config"):
        raise ValueError(f"Config {config.name!r} does not expose data.base_config")

    base_config = dataclasses.replace(config.data.base_config, lerobot_video_backend=backend)
    data_config = dataclasses.replace(config.data, base_config=base_config)
    return dataclasses.replace(config, data=data_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name")
    parser.add_argument("--backend", default=None, choices=("pyav", "torchcodec", "video_reader"))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--skip-norm-stats", action="store_true")
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    config = _override_video_backend(config, args.backend)
    if args.num_workers is not None:
        config = dataclasses.replace(config, num_workers=args.num_workers)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)

    print(
        "config",
        config.name,
        "batch_size",
        config.batch_size,
        "num_workers",
        config.num_workers,
        "backend",
        getattr(config.data.base_config, "lerobot_video_backend", None),
        flush=True,
    )

    start = time.time()
    loader = _data_loader.create_data_loader(
        config,
        shuffle=args.shuffle,
        num_batches=args.batches,
        skip_norm_stats=args.skip_norm_stats,
        framework="pytorch",
    )
    print("loader_init_s", round(time.time() - start, 3), flush=True)

    iterator = iter(loader)
    batch_times: list[float] = []
    for batch_index in range(args.batches):
        batch_start = time.time()
        _observation, actions = next(iterator)
        elapsed = time.time() - batch_start
        batch_times.append(elapsed)
        print(
            "batch",
            batch_index,
            "dt_s",
            round(elapsed, 3),
            "actions_shape",
            tuple(actions.shape),
            flush=True,
        )

    total = time.time() - start
    avg = sum(batch_times) / len(batch_times)
    print("summary", "total_s", round(total, 3), "avg_batch_s", round(avg, 3), flush=True)


if __name__ == "__main__":
    main()
