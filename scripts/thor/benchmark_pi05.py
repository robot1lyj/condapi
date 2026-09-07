"""Run one precision configuration on a recorded YAM observation, without robot IO."""

import argparse
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import jax
import numpy as np

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_observation(path):
    with np.load(path, allow_pickle=False) as data:
        observation = {key: np.array(data[key]) for key in data.files}
    state = observation["observation.state"]
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Recorded state must be finite 14D")
    for view in ("top_rgb", "left_rgb", "right_rgb"):
        image = observation[f"observation.images.{view}"]
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Recorded images must be HWC uint8 RGB")
    observation["prompt"] = str(observation["prompt"].item())
    if not observation["prompt"].strip():
        raise ValueError("Recorded observation needs a task prompt")
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--config", default="pi05_yam")
    parser.add_argument("--params-dtype", choices=("checkpoint", "bfloat16", "float32"), default="checkpoint")
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats < 2:
        parser.error("steps must be positive and repeats at least two")
    if args.output.exists() or args.output.with_suffix(".actions.npy").exists():
        parser.error("Use a new output path; never overwrite a previous experiment")
    if (args.checkpoint / "model.safetensors").exists():
        parser.error("This benchmark is JAX-only")
    provenance = json.loads(args.provenance.read_text())
    if provenance.get("source_kind") != "real_yam_recording" or not provenance.get("source_files"):
        parser.error("Real recorded sample provenance is required; no synthetic fallback")
    if provenance.get("sample_sha256") != digest(args.sample):
        parser.error("Recorded sample digest does not match provenance")
    if provenance.get("norm_stats_sha256") != digest(args.norm_stats):
        parser.error("Normalization stats do not match sample provenance")
    if any(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError("Thor GPU execution required")
    jax.config.update("jax_default_matmul_precision", "highest")
    observation = read_observation(args.sample)
    train_config = config.get_config(args.config)
    if (
        not isinstance(train_config.data, config.LeRobotYamDataConfig)
        or not train_config.model.pi05
        or train_config.model.action_dim != 32
        or train_config.model.action_horizon != 50
    ):
        parser.error("This replay requires the Pi0.5 YAM 32D/50-step configuration")
    train_config = dataclasses.replace(
        train_config, model=dataclasses.replace(train_config.model, dtype=args.compute_dtype)
    )
    norm_stats = normalize.deserialize_json(args.norm_stats.read_text())
    started = time.monotonic()
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        norm_stats=norm_stats,
        sample_kwargs={"num_steps": args.steps},
        jax_param_dtype=args.params_dtype,
    )
    load_s = time.monotonic() - started
    noise = np.random.default_rng(args.seed).standard_normal((50, 32)).astype(np.float32)
    latencies = []
    outputs = []
    first_s = None
    for index in range(args.repeats + 2):
        started = time.monotonic()
        # policy.infer materializes NumPy outputs: this measures completed execution,
        # transforms and host transfer, not merely an asynchronous JAX dispatch.
        action = np.asarray(policy.infer(observation, noise=noise)["actions"], dtype=np.float32)
        elapsed = time.monotonic() - started
        if action.shape != (50, 14) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid YAM output shape/values: {action.shape}")
        if index == 0:
            first_s = elapsed
        if index >= 2:
            latencies.append(elapsed * 1000)
            outputs.append(action)
    actions = np.stack(outputs)
    record = {
        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "measured_not_accuracy_approved",
        "scope": "real_yam_offline_replay_no_robot_execution",
        "config": args.config,
        "params_dtype": args.params_dtype,
        "compute_dtype": args.compute_dtype,
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata_sha256": digest(args.checkpoint / "params" / "_METADATA"),
        "sample_sha256": digest(args.sample),
        "norm_stats_sha256": digest(args.norm_stats),
        "provenance": provenance,
        "seed": args.seed,
        "steps": args.steps,
        "repeats": args.repeats,
        "load_s": load_s,
        "first_infer_s": first_s,
        "latencies_ms": latencies,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "repeat_max_abs_difference": float(np.max(np.abs(actions - actions[0]))),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "versions": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "flax", "orbax-checkpoint")},
        "action_shape": list(actions.shape),
        "limitations": [
            "Base checkpoint is not YAM-finetuned",
            "Not a robot task success-rate test",
            "Process RSS is not total GPU/unified memory usage",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".actions.npy"), actions, allow_pickle=False)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
