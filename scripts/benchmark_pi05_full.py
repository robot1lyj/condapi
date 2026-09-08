"""Bounded full-parameter Pi0.5 capacity probe with real base weights.

This measures memory/compute feasibility, not dataset correctness or convergence.
Run within a GPU allocation, with XLA_PYTHON_CLIENT_PREALLOCATE=false for useful
allocator high-water marks. No checkpoint or online telemetry is produced.
"""

import dataclasses
import functools
import json
import pathlib
import time

import jax
import numpy as np
import train
import tyro

from openpi.models import model as model_lib
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import sharding
from openpi.training import weight_loaders


def main(
    checkpoint: str,
    output: pathlib.Path,
    batch_size: int = 4,
    steps: int = 3,
    dataset: str | None = None,
    assets: str | None = None,
):
    if steps < 2 or batch_size < 1 or batch_size % jax.device_count():
        raise ValueError("Need >=2 steps and a positive batch divisible by the device count")
    output.mkdir(parents=True, exist_ok=False)
    config = dataclasses.replace(
        config_lib.get_config("pi05_yam"),
        batch_size=batch_size,
        fsdp_devices=jax.device_count(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(checkpoint),
    )
    if dataset is not None:
        if assets is None:
            raise ValueError("Real dataset requires its independently computed norm assets")
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                repo_id=dataset,
                assets=config_lib.AssetsConfig(assets_dir=assets, asset_id="yam"),
                base_config=dataclasses.replace(config.data.base_config, train_episodes=(0,)),
            ),
        )
    mesh = sharding.make_mesh(config.fsdp_devices)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    def report(stage, **values):
        record = {
            "stage": stage,
            "time": time.time(),
            "devices": [{"device": str(d), "memory": d.memory_stats()} for d in jax.devices()],
            **values,
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    report(
        "start",
        batch_size=batch_size,
        fsdp_devices=config.fsdp_devices,
        ema=None,
        input=dataset or "synthetic",
        assets=assets,
        checkpoint=checkpoint,
        model=str(config.model),
    )
    started = time.monotonic()
    state, state_sharding = train.init_train_state(config, jax.random.key(0), mesh, resume=False)
    jax.block_until_ready(state)
    total = sum(x.size for x in jax.tree.leaves(state.params))
    trainable = sum(x.size for x in jax.tree.leaves(state.params.filter(config.trainable_filter)))
    if total != trainable:
        raise ValueError(f"Not full tuning: {trainable}/{total}")
    report("initialized", seconds=time.monotonic() - started, parameters=total, trainable_parameters=trainable)
    if dataset is None:
        source = data_loader.FakeDataset(config.model, batch_size)
    else:
        data_config = config.data.create(config.assets_dirs, config.model)
        source = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        source = data_loader.transform_dataset(source, data_config)
    rows = [source[i] for i in range(batch_size)]
    packed = jax.tree.map(lambda *xs: np.stack(xs), *rows)
    actions = packed.pop("actions")
    if dataset is None:
        packed["image_mask"] = jax.tree.map(np.ones_like, packed["image_mask"])
        packed["tokenized_prompt_mask"] = np.ones_like(packed["tokenized_prompt_mask"])
    batch = jax.device_put((model_lib.Observation.from_dict(packed), actions), data_sharding)
    report("batch", action_shape=list(actions.shape), state_shape=list(packed["state"].shape))
    step_fn = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    for step in range(steps):
        started = time.monotonic()
        with sharding.set_mesh(mesh):
            state, info = step_fn(jax.random.key(step + 1), state, batch)
        jax.block_until_ready((state, info))
        values = {k: float(v) for k, v in jax.device_get(info).items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError(f"Nonfinite metrics: {values}")
        report("step", step=step, seconds=time.monotonic() - started, **values)
    report("complete", optimizer_steps=int(state.step))


if __name__ == "__main__":
    tyro.cli(main)
