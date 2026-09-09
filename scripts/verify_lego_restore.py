"""Read a full-size benchmark checkpoint, perform one update, never overwrite it."""

import functools
import json

import jax
import numpy as np
import train
from train_lego_full import make_config

from openpi.training import checkpoints
from openpi.training import data_loader
from openpi.training import sharding


def main():
    config = make_config()
    mesh = sharding.make_mesh(4)
    state, state_sharding = train.init_train_state(config, jax.random.key(42), mesh, resume=True)
    source = "/home/wuyan/lyj/YAM/env-transfer/pi05-batch-limit-20260908.eLaflYGQ/b64-stream/checkpoint"
    manager, resuming = checkpoints.initialize_checkpoint_dir(source, keep_period=None, overwrite=False, resume=True)
    if not resuming:
        raise RuntimeError("Expected the existing full-size probe checkpoint")
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True, num_batches=1)
    state = checkpoints.restore_state(manager, state, state_sharding, loader)
    if int(state.step) != 30:
        raise RuntimeError(f"Unexpected restored step {state.step}")
    loader.set_start_step(30)
    batch = next(iter(loader))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    update = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    with sharding.set_mesh(mesh):
        state, info = update(jax.random.key(42), state, batch)
    jax.block_until_ready((state, info))
    values = {key: float(value) for key, value in jax.device_get(info).items()}
    if int(state.step) != 31 or not all(np.isfinite(value) for value in values.values()):
        raise RuntimeError("Restored update failed")
    print(
        json.dumps({"restore_verified": True, "source": source, "step_before": 30, "step_after": 31, **values}),
        flush=True,
    )
    manager.close()


if __name__ == "__main__":
    main()
