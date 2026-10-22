"""Opt-in per-update numerical forensics for the native OpenPI trainer.

No training loop lives here. Device summaries inspect actual addressable shards;
the native loop decides whether it is safe to call its optimizer update.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import numerical_evidence as evidence


def local_statistics(arrays):
    """One small result per local tensor; never all-gather a full model leaf."""
    integers, floats = [], []
    for array in arrays:
        if jnp.issubdtype(array.dtype, jnp.complexfloating) or str(array.dtype) == "float64":
            raise ValueError("Shard evidence supports real <=FP32 tensors; refusing a lossy diagnostic cast")
        flat = array.astype(jnp.float32).reshape(-1)
        if not flat.size:
            integers.append(jnp.array([0, 0, 0, 0, -1, -1], jnp.int32))
            floats.append(jnp.zeros(4, jnp.float32))
            continue
        finite = jnp.isfinite(flat)
        clean = jnp.where(finite, flat, 0)
        absolute = jnp.abs(clean)
        maximum = jnp.max(absolute)
        denominator = jnp.where(maximum > 0, maximum, 1)
        scaled_sum = jnp.sum(jnp.square(clean / denominator), dtype=jnp.float32)
        first_bad = jnp.argmax(~finite)
        max_index = jnp.argmax(absolute)
        all_finite = jnp.all(finite)
        integers.append(
            jnp.stack(
                (
                    jnp.asarray(flat.size, jnp.int32),
                    jnp.sum(jnp.isnan(flat)),
                    jnp.sum(jnp.isposinf(flat)),
                    jnp.sum(jnp.isneginf(flat)),
                    jnp.where(all_finite, -1, first_bad),
                    max_index,
                )
            )
        )
        floats.append(jnp.stack((maximum, scaled_sum, jnp.where(all_finite, 0, flat[first_bad]), flat[max_index])))
    return jnp.stack(integers), jnp.stack(floats)


def index_json(index):
    return [[part.start, part.stop, part.step] if isinstance(part, slice) else int(part) for part in index]


class NumericalDiagnostics:
    def __init__(self, directory, config):
        if jax.process_count() != 1:
            raise ValueError("Numerical evidence currently requires one process with all devices addressable")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.journal = (self.directory / "steps.jsonl").open("x", buffering=1)
        self.statistics = jax.jit(local_statistics)
        self.schemas = {}
        self.failure_written = False
        self.context = {}
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
        evidence.write_json(
            self.directory / "manifest.json",
            {
                "version": 1,
                "created_utc": evidence.utc_now(),
                "code_commit": commit,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "run": config.exp_name,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "fsdp_devices": config.fsdp_devices,
                "checkpoint_directory": str(config.checkpoint_dir),
                "dataset": config.data.repo_id,
                "mode": "split_native_gradient_and_optimizer_with_host_guard",
                "integer_fields": evidence.INTEGER_FIELDS,
                "float_fields": evidence.FLOAT_FIELDS,
                "devices": [
                    {
                        "id": d.id,
                        "process_index": d.process_index,
                        "kind": d.device_kind,
                        "local_hardware_id": getattr(d, "local_hardware_id", None),
                    }
                    for d in jax.devices()
                ],
                "limitations": [
                    "Instrumentation changes compilation and adds synchronization",
                    "Post-update failures retain post-state; pre-state buffers may have been donated",
                    "A GPU/native crash can prevent tensor export; external incident capture remains active",
                ],
            },
        )

    def record(self, event, **fields):
        record = {"utc": evidence.utc_now(), "event": event, **self.context, **fields}
        self.journal.write(json.dumps(evidence.json_safe(record), separators=(",", ":"), allow_nan=False) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def begin(self, step, identity, loader_hash, rng):
        self.context = {
            "attempted_completed_step": step + 1,
            "pre_update_step": step,
            "batch_identity": identity,
            "loader_tensor_sha256": loader_hash,
            "loader_hash_stage": "before_Observation.from_dict",
            "step_rng": np.asarray(jax.device_get(jax.random.key_data(jax.random.fold_in(rng, step)))),
        }
        self.record("attempt_started")

    def inspect(self, tree, label):
        by_device = {}
        for path, array in jax.tree_util.tree_flatten_with_path(tree)[0]:
            if not hasattr(array, "addressable_shards"):
                continue
            for shard in array.addressable_shards:
                item = {
                    "path": jax.tree_util.keystr(path),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "device_id": shard.device.id,
                    "index": index_json(shard.index),
                    "local_shape": list(shard.data.shape),
                    "replica_id": shard.replica_id,
                }
                by_device.setdefault(shard.device.id, []).append((item, shard.data))
        pending = []
        for device_id in sorted(by_device):
            group = by_device[device_id]
            pending.append(([item for item, _ in group], self.statistics(tuple(array for _, array in group))))
        records, schema = [], []
        for descriptions, values in pending:
            integers, floats = jax.device_get(values)
            for description, counts, numbers in zip(descriptions, integers, floats, strict=True):
                schema.append(description)
                records.append({**description, "integers": counts.tolist(), "floats": numbers.tolist()})
        if not records:
            raise RuntimeError(f"No device arrays found for numerical inspection: {label}")
        if label not in self.schemas:
            evidence.write_json(self.directory / ("schema-" + label + ".json"), schema)
            self.schemas[label] = schema
        elif self.schemas[label] != schema:
            raise RuntimeError(f"Numerical evidence layout changed for {label}")
        summary = evidence.summarize_records(records)
        summary["bad_shards"] = [records[index] for index in summary["bad_record_indexes"]]
        return {
            "schema": label,
            "summary": summary,
            "integers": [r["integers"] for r in records],
            "floats": [r["floats"] for r in records],
        }

    def failure(self, reason, *, state, batch, rng, gradients=None, updates=None, phase="pre_update", details=None):
        """Publish the failure first, then preserve actual tensors shard by shard.

        These files are a forensic bundle, never a new official recovery point.
        Partial exports remain present with explicit errors and no COMPLETE marker.
        """
        if self.failure_written:
            return
        self.failure_written = True
        destination = self.directory / f"failure-step-{self.context.get('attempted_completed_step', 'initial')}"
        destination.mkdir(mode=0o700)
        header = {
            "utc": evidence.utc_now(),
            **self.context,
            "reason": reason,
            "phase": phase,
            "details": details,
            "official_checkpoint": False,
            "state_semantics": "pre-update (optimizer not called)"
            if phase == "pre_update"
            else "post-update or runtime failure; see phase, do not assume pre-update",
            "tensor_export_complete": False,
        }
        evidence.write_json(destination / "failure.json", header)
        self.record("failure_detected", reason=reason, phase=phase, directory=str(destination))
        (destination / "NOT_A_TRAINING_CHECKPOINT").write_text("Forensic tensors only. Do not resume automatically.\n")
        payload = {
            "batch": batch,
            "rng": jax.random.key_data(rng),
            "state": {
                "params": state.params,
                "optimizer": state.opt_state,
                "step": state.step,
                "ema_params": state.ema_params,
            },
        }
        if gradients is not None:
            payload["gradients"] = gradients
        if updates is not None:
            payload["optimizer_updates"] = updates
        records = []
        complete = True
        for leaf_index, (path, array) in enumerate(jax.tree_util.tree_flatten_with_path(payload)[0]):
            path_string = jax.tree_util.keystr(path)
            if hasattr(array, "addressable_shards"):
                shards = [(shard.device.id, index_json(shard.index), shard.data) for shard in array.addressable_shards]
            else:
                shards = [(None, [], array)]
            for shard_number, (device_id, index, value) in enumerate(shards):
                record = {
                    "path": path_string,
                    "device_id": device_id,
                    "index": index,
                    "global_shape": list(getattr(array, "shape", ())),
                    "dtype": str(getattr(array, "dtype", type(array))),
                }
                filename = f"tensor-{leaf_index:05d}-shard-{shard_number:02d}.npy"
                try:
                    host = np.asarray(jax.device_get(value))
                    if str(host.dtype) == "bfloat16":
                        host = host.view(np.uint16)
                        record["storage_encoding"] = "bfloat16_bits_as_uint16"
                    with (destination / filename).open("xb") as handle:
                        np.save(handle, host, allow_pickle=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                    digest = hashlib.sha256()
                    with (destination / filename).open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    record.update(
                        file=filename,
                        bytes=(destination / filename).stat().st_size,
                        sha256=digest.hexdigest(),
                        status="saved",
                    )
                except Exception as exc:
                    complete = False
                    record.update(status="failed", error=repr(exc))
                records.append(record)
                evidence.write_json(destination / "tensor-manifest.json", records)
        header["tensor_export_complete"] = complete
        evidence.write_json(destination / "failure.json", header)
        if complete:
            (destination / "COMPLETE").write_text(evidence.utc_now() + "\n")
        self.record("failure_export_finished", complete=complete, tensors=len(records))

    def update(self, state, batch, rng, gradient_function, update_function):
        phase = "input"
        gradients = None
        updates = None
        try:
            observed_step = int(jax.device_get(state.step))
            if observed_step != self.context["pre_update_step"]:
                raise RuntimeError(f"Train-state step mismatch: {observed_step} != {self.context['pre_update_step']}")
            inputs = self.inspect(batch, "input")
            self.record("input_checked", statistics=inputs)
            if not inputs["summary"]["all_finite"]:
                self.failure("nonfinite_input", state=state, batch=batch, rng=rng, details=inputs)
                raise FloatingPointError("Nonfinite input; optimizer was not called")
            phase = "gradient"
            loss, gradients, native_norm = gradient_function(rng, state, batch)
            statistics = self.inspect(gradients, "gradient")
            host_loss, host_norm = jax.device_get((loss, native_norm))
            verdict = evidence.gradient_verdict(host_loss, host_norm, statistics["summary"])
            stable_norm = statistics["summary"]["stable_finite_values_l2"]
            verdict["native_vs_stable_relative_difference"] = (
                abs(float(host_norm) - stable_norm) / max(stable_norm, 1e-30)
                if math.isfinite(float(host_norm))
                else None
            )
            self.record(
                "gradients_checked", loss=host_loss, native_grad_norm=host_norm, statistics=statistics, verdict=verdict
            )
            if not verdict["update_allowed"]:
                self.failure(
                    "gradient_guard",
                    state=state,
                    batch=batch,
                    rng=rng,
                    gradients=gradients,
                    details={
                        "loss": host_loss,
                        "native_grad_norm": host_norm,
                        "statistics": statistics,
                        "verdict": verdict,
                    },
                )
                raise FloatingPointError("Numerical gradient guard stopped BEFORE optimizer update; evidence saved")
            phase = "optimizer"
            # Healthy state/gradients may be donated here. On a later failure we
            # explicitly label the exported state as post-update, never pre-update.
            state, param_norm, updates = update_function(state, gradients)
            gradients = None
            phase = "post_update"
            observed_step = int(jax.device_get(state.step))
            if observed_step != self.context["attempted_completed_step"]:
                raise RuntimeError(f"Updated train-state step mismatch: {observed_step}")
            update_stats = self.inspect(updates, "optimizer_updates")
            updated = self.inspect({"params": state.params, "optimizer": state.opt_state}, "state")
            host_param_norm = jax.device_get(param_norm)
            self.record(
                "update_finished", statistics=updated, optimizer_updates=update_stats, param_norm=host_param_norm
            )
            if (
                not updated["summary"]["all_finite"]
                or not update_stats["summary"]["all_finite"]
                or not math.isfinite(float(host_param_norm))
            ):
                self.failure(
                    "nonfinite_updated_state",
                    state=state,
                    batch=batch,
                    rng=rng,
                    updates=updates,
                    phase=phase,
                    details={"state": updated, "updates": update_stats, "param_norm": host_param_norm},
                )
                raise FloatingPointError("Nonfinite parameter/optimizer after update; forensic state saved")
            return state, {"loss": loss, "grad_norm": native_norm, "param_norm": param_norm}
        except Exception as exc:
            if not self.failure_written:
                self.record("runtime_error", phase=phase, exception=repr(exc))
                try:
                    self.failure(
                        "runtime_error",
                        state=state,
                        batch=batch,
                        rng=rng,
                        gradients=gradients,
                        updates=updates,
                        phase=phase,
                        details={"exception": repr(exc)},
                    )
                except Exception as export_error:
                    self.record("failure_export_error", phase=phase, exception=repr(export_error))
            raise

    def close(self):
        self.journal.close()
