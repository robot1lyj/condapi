"""Host-side numerical evidence helpers; no framework/device initialization."""

from datetime import UTC
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

IDENTITY_KEY = "_numeric_identity"
INTEGER_FIELDS = ("size", "nan_count", "posinf_count", "neginf_count", "first_bad_flat_index", "max_abs_flat_index")
FLOAT_FIELDS = ("max_abs_finite", "scaled_sum_squares", "first_bad_value", "max_abs_value")


def utc_now():
    return datetime.now(UTC).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
    return value


def write_json(path, value):
    """Publish one complete JSON file atomically; callers use a fresh directory."""
    path = Path(path)
    temp = path.with_name(path.name + ".writing")
    with temp.open("x") as handle:
        json.dump(json_safe(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def sample_identity(sample, index):
    """Keep LeRobot identities before repack/model transforms discard them."""
    if IDENTITY_KEY in sample:
        return sample[IDENTITY_KEY]
    result = {"dataset_index": np.asarray(int(index), dtype=np.int64)}
    for name in ("index", "episode_index", "frame_index", "task_index"):
        value = np.asarray(sample.get(name, -1))
        result[name] = np.asarray(int(value.reshape(-1)[0]), dtype=np.int64)
    timestamp = np.asarray(sample.get("timestamp", np.nan))
    result["timestamp"] = np.asarray(float(timestamp.reshape(-1)[0]), dtype=np.float64)
    return result


def batch_fingerprint(batch):
    """Hash actual loader tensors on the host before Observation.from_dict."""
    digest = hashlib.sha256()

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], (*path, str(key)))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))
        elif value is not None:
            array = np.ascontiguousarray(np.asarray(value))
            if array.dtype.hasobject:
                raise TypeError(f"Cannot fingerprint object dtype at {path}")
            digest.update(json.dumps([path, str(array.dtype), array.shape], separators=(",", ":")).encode())
            digest.update(array.tobytes())

    visit(batch, ())
    return digest.hexdigest()


def summarize_records(records):
    """Stable float64 host norm, counting a replicated shard only once.

    Per-device entries remain intact so a bad replica cannot be hidden by the
    representative used for the logical norm. This is evidence, not an optimizer.
    """
    seen = set()
    norm = 0.0
    bad = []
    replicas = {}
    disagreements = []
    for index, record in enumerate(records):
        counts = record["integers"]
        values = record["floats"]
        if any(counts[1:4]):
            bad.append(index)
        key = (record["path"], json.dumps(record["index"], sort_keys=True))
        signature = (tuple(counts), tuple(json_safe(values)))
        if key in replicas and signature != replicas[key]:
            disagreements.append(index)
        replicas[key] = signature
        if key not in seen:
            seen.add(key)
            norm = math.hypot(norm, float(values[0]) * math.sqrt(max(0.0, float(values[1]))))
    return {
        "all_finite": not bad,
        "bad_record_indexes": bad,
        "replica_disagreement_indexes": disagreements,
        "stable_finite_values_l2": norm,
        "logical_shards": len(seen),
        "physical_shards": len(records),
    }


def gradient_verdict(loss, native_norm, summary):
    reasons = []
    if not math.isfinite(float(loss)):
        reasons.append("nonfinite_loss")
    if not summary["all_finite"]:
        reasons.append("nonfinite_gradient_elements")
    if not math.isfinite(float(native_norm)):
        reasons.append("nonfinite_native_gradient_norm")
    if not math.isfinite(summary["stable_finite_values_l2"]):
        reasons.append("nonfinite_stable_gradient_norm")
    return {
        "update_allowed": not reasons,
        "reasons": reasons,
        "finite_elements_but_native_norm_nonfinite": summary["all_finite"] and not math.isfinite(float(native_norm)),
    }
