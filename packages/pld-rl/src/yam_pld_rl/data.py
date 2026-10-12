"""Pure-array contracts for immutable, physical-time replay preparation."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def physical_edges(times, ticks, epochs, action_valid, max_dt=0.15):
    """Camera loss is not physical execution loss. Never compact the time axis."""
    dt = np.diff(times)
    return (
        action_valid[:-1]
        & action_valid[1:]
        & (np.diff(ticks) == 1)
        & (np.diff(epochs) == 0)
        & np.isfinite(dt)
        & (dt > 0)
        & (dt <= max_dt)
    )


def observed_edges(physical, valid):
    return physical & valid[:-1] & valid[1:]


def suffix_return(times, connected, terminal_row, reward, discount_per_second):
    """MC calibration exists only on the physically connected terminal suffix.

    A temporary missing camera does not erase an executed action or known
    outcome. It still removes observation transitions via observed_edges.
    A tick, epoch, time or action gap DOES invalidate earlier return calibration.
    """
    if not 0 < discount_per_second <= 1:
        raise ValueError("discount_per_second must be in (0,1]")
    result = np.full(len(times), np.nan, dtype=np.float64)
    # Reward belongs to the last transition, so Q at terminal_row-1 equals r.
    for row in range(terminal_row - 1, -1, -1):
        if not connected[row]:
            break
        remaining = times[terminal_row - 1] - times[row]
        result[row] = reward * discount_per_second**remaining
    return result


def action_coordinates(action, state):
    """Relative joint TARGETS, absolute grippers; no unit conversions."""
    value = np.array(action, dtype=np.float32, copy=True)
    value[..., :6] -= state[..., :6]
    value[..., 7:13] -= state[..., 7:13]
    return value


def fit_normalization(state, action, train_mask):
    if not np.any(train_mask):
        raise ValueError("no training rows")
    return {
        "state_mean": state[train_mask].mean(0).tolist(),
        "state_std": np.maximum(state[train_mask].std(0), 1e-3).tolist(),
        "action_mean": action[train_mask].mean(0).tolist(),
        "action_std": np.maximum(action[train_mask].std(0), 1e-3).tolist(),
        "fit_split": "E02-E14 only, both outcomes; shared by all experiments",
    }
