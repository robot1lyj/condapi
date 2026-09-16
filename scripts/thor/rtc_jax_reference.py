"""Generate original-JAX trained-RTC references for recorded aligned YAM cases."""

import argparse
import dataclasses
import json
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
from benchmark_suite import checked_path
import jax
import jax.numpy as jnp
import numpy as np
from rtc_action_space import encode_committed_actions
from rtc_provenance import source_params_sha256

from openpi.models import model as model_api
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.checkpoint / "model.safetensors").exists():
        parser.error("Use a new output directory and the original JAX checkpoint")
    if any(device.platform != "gpu" for device in jax.devices()):
        parser.error("Original JAX reference requires Thor GPU")
    contract = json.loads(args.training_contract.read_text())
    max_delay = contract["model"]["rtc_training_max_delay"]
    if not 0 < max_delay < 50:
        parser.error("Original checkpoint must have trained RTC delay")
    norm_path = args.checkpoint / "assets" / "yam" / "norm_stats.json"
    if digest(norm_path) != contract["norm_sha256"]:
        raise ValueError("JAX checkpoint norm differs from training contract")
    case_set = json.loads(args.cases.read_text())
    if case_set.get("source_kind") != "real_yam_recording" or case_set.get("norm_stats_sha256") != digest(norm_path):
        raise ValueError("RTC reference needs real cases and matching norm")
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(
        train.model, dtype="float32", rtc_training_max_delay=max_delay,
    ))
    stats = normalize.deserialize_json(norm_path.read_text())
    policy = policy_config.create_trained_policy(
        train, args.checkpoint, norm_stats=stats, jax_param_dtype="checkpoint",
    )
    noise = np.random.default_rng(0).standard_normal((1, 50, 32)).astype(np.float32)
    args.output.mkdir(parents=True)
    results = []
    for index, row in enumerate(case_set["cases"]):
        sample_path = checked_path(args.cases.parent, row["sample"])
        provenance = json.loads(checked_path(args.cases.parent, row["provenance"]).read_text())
        prefix_path = checked_path(args.cases.parent, row["committed_actions"])
        if (
            provenance.get("source_kind") != "real_yam_recording"
            or digest(sample_path) != provenance.get("sample_sha256")
            or provenance.get("norm_stats_sha256") != digest(norm_path)
            or digest(prefix_path) != row["committed_actions_sha256"]
            or row.get("source_episode") not in provenance.get("source_files", {})
        ):
            raise ValueError("RTC JAX case provenance mismatch")
        delay = row["delay_steps"]
        if not 0 <= delay <= max_delay or row["target_start_tick"] != row["committed_start_tick"]:
            raise ValueError("RTC JAX case has invalid delay or tick alignment")
        physical_prefix = np.load(prefix_path, allow_pickle=False)
        if physical_prefix.shape != (delay, 14) or not np.isfinite(physical_prefix).all():
            raise ValueError("RTC JAX committed prefix must be finite [delay,14]")
        observation = read_observation(sample_path)
        state = observation["observation.state"]
        absolute = np.broadcast_to(state, (50, 14)).copy()
        absolute[:delay] = physical_prefix
        previous = encode_committed_actions(absolute, state, stats)
        transformed = policy._input_transform(dict(observation))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], transformed)
        model_obs = model_api.Observation.from_dict(inputs)
        raw = policy._model.sample_actions_trained_rtc(  # noqa: SLF001
            jax.random.key(0), model_obs, previous_actions=jnp.asarray(previous[None]),
            delay_steps=delay, num_steps=10, noise=jnp.asarray(noise),
        )
        raw = np.asarray(raw[0], dtype=np.float32)
        outputs = policy._output_transform({"state": np.asarray(inputs["state"][0]), "actions": raw})  # noqa: SLF001
        physical = np.asarray(outputs["actions"], dtype=np.float32)
        physical[:delay] = physical_prefix
        if raw.shape != (50, 32) or physical.shape != (50, 14) or not np.isfinite(raw).all():
            raise RuntimeError("Invalid JAX RTC reference output")
        path = args.output / f"case-{index:03d}.npz"
        np.savez_compressed(path, normalized=raw, physical=physical)
        results.append({"sample": row["sample"], "delay_steps": delay, "reference": path.name, "sha256": digest(path)})
    (args.output / "reference_manifest.json").write_text(json.dumps({
        "status": "original_jax_trained_rtc_reference",
        "checkpoint_metadata_sha256": digest(args.checkpoint / "params" / "_METADATA"),
        "source_params_files_sha256": source_params_sha256(args.checkpoint),
        "training_contract_sha256": digest(args.training_contract),
        "norm_stats_sha256": digest(norm_path),
        "cases_sha256": digest(args.cases),
        "noise_seed": 0,
        "precision": "FP32 original checkpoint/compute",
        "cases": results,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
