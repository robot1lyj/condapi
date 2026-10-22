#!/usr/bin/env python3
"""Allocated-GPU evidence self-check using tiny arrays, no model or optimizer."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.training import numerical_evidence as evidence
from openpi.training.numerical_diagnostics import NumericalDiagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices = jax.devices()
    if len(devices) != 4 or any(d.platform != "gpu" for d in devices):
        raise RuntimeError("Self-check requires the allocated four GPUs")
    config = SimpleNamespace(
        exp_name="synthetic-numerical-evidence-check",
        seed=42,
        batch_size=4,
        fsdp_devices=4,
        checkpoint_dir="not-a-checkpoint",
        data=SimpleNamespace(repo_id="synthetic"),
    )
    diagnostic = NumericalDiagnostics(args.output, config)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("check",))
    spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("check"))
    blocks = [
        np.array([3, 4], np.float32),
        np.array([np.nan, np.inf], np.float32),
        np.array([1e30, -1e30], np.float32),
        np.array([0, 0], np.float32),
    ]
    array = jax.make_array_from_single_device_arrays(
        (8,), spec, [jax.device_put(block, device) for block, device in zip(blocks, devices, strict=True)]
    )
    result = diagnostic.inspect({"test_gradient": array}, "gradient")
    assert not result["summary"]["all_finite"]
    assert len(result["summary"]["bad_shards"]) == 1
    bad = result["summary"]["bad_shards"][0]
    assert bad["device_id"] == devices[1].id
    assert bad["integers"][1:4] == [1, 1, 0]
    assert bad["index"] == [[2, 4, None]]
    # Exercise the exact kind of buffers used by a resumed native trainer,
    # rather than relying exclusively on explicitly committed device_put arrays.
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpoint_path = args.output.resolve() / "synthetic-orbax-state"
        checkpointer.save(checkpoint_path, {"restored_shards": array})
        restored_state = checkpointer.restore(
            checkpoint_path,
            item={"restored_shards": jax.ShapeDtypeStruct(array.shape, array.dtype, sharding=spec)},
            restore_args={"restored_shards": ocp.ArrayRestoreArgs(restore_type=jax.Array, sharding=spec)},
        )
    restored_stats = diagnostic.inspect(restored_state, "orbax_restored")
    assert restored_stats["summary"]["bad_shards"][0]["device_id"] == devices[1].id
    assert restored_stats["summary"]["bad_shards"][0]["integers"][1:4] == [1, 1, 0]
    assert np.isclose(result["summary"]["stable_finite_values_l2"], np.sqrt(2) * 1e30, rtol=1e-6)
    assert not evidence.gradient_verdict(0.01, float("inf"), result["summary"])["update_allowed"]
    large = jax.device_put(np.array([1e30, -1e30], np.float32), devices[0])
    large_result = diagnostic.inspect({"large": large}, "finite_large")
    naive_norm = jax.jit(lambda x: jnp.sqrt(jnp.sum(jnp.square(x))))(large)
    assert np.isinf(float(jax.device_get(naive_norm)))
    assert large_result["summary"]["all_finite"]
    assert evidence.gradient_verdict(0.01, float(naive_norm), large_result["summary"])[
        "finite_elements_but_native_norm_nonfinite"
    ]
    replicated = jax.device_put(
        np.array([3, 4], np.float32), jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    )
    replicated_result = diagnostic.inspect({"replica": replicated}, "replicated")
    assert replicated_result["summary"]["physical_shards"] == 4
    assert replicated_result["summary"]["logical_shards"] == 1
    assert replicated_result["summary"]["stable_finite_values_l2"] == 5
    rng = jax.random.key(42)
    state = SimpleNamespace(
        params={"sample": jax.device_put(np.array([1.0], np.float32), devices[0])},
        opt_state={},
        step=1000,
        ema_params=None,
    )
    diagnostic.begin(1000, {"dataset_index": [1, 2, 3, 4]}, "synthetic", rng)
    diagnostic.failure(
        "intentional_synthetic_array_check",
        state=state,
        batch={"sample": array},
        rng=rng,
        gradients={"test_gradient": array},
        details=result,
    )
    folder = args.output / "failure-step-1001"
    assert (folder / "COMPLETE").exists()
    files = json.loads((folder / "tensor-manifest.json").read_text())
    assert all(r["status"] == "saved" for r in files)
    # Read the actual saved offending shard back, preserving NaN/+Inf exactly.
    saved = next(r for r in files if "gradients" in r["path"] and r["device_id"] == devices[1].id)
    restored = np.load(folder / saved["file"], allow_pickle=False)
    assert np.isnan(restored[0])
    assert np.isposinf(restored[1])
    diagnostic.close()
    evidence.write_json(
        args.output / "SELF_CHECK_PASSED.json",
        {
            "synthetic_arrays_only": True,
            "model_or_optimizer_executed": False,
            "devices": [d.id for d in devices],
            "correct_bad_device": bad["device_id"],
            "shard_metadata_and_tensor_readback": "passed",
            "stable_large_value_norm": "passed",
            "orbax_restored_shard_placement": "passed",
        },
    )
    print("NUMERICAL_EVIDENCE_SELF_CHECK=PASS", flush=True)


if __name__ == "__main__":
    main()
