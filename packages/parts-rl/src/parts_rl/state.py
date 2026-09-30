"""Explicit queue-aware state encoding shared by inference and replay."""

import numpy as np

PHASES = ("READY", "ACTIVE_DESCENT", "ACTIVE_CLOSURE", "EXIT_PENDING", "WAIT_REARM")
STATE_SCHEMA = "yam-parts-state-v1"
NON_VISUAL_DIM = 14 + 50 * 14 + 8 + len(PHASES) + 50 * 14 + 50 + 50
REFERENCE_SLICE = slice(14, 14 + 50 * 14)


def array(value, shape, name):
    out = np.asarray(value, dtype=np.float32)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite {shape}")
    return out


def encode_state(z, observation_state, actions, parts, arm, *, feature_dim):
    visual = array(z, (feature_dim,), "visual features")
    feedback = array(observation_state, (14,), "state")
    reference = array(actions, (50, 14), "actions")
    state = parts["arms"][arm]
    force_valid = state.get("force_valid")
    if type(force_valid) is not bool or type(state.get("eligible")) is not bool:
        raise ValueError("force_valid/eligible must be explicit booleans")
    if state.get("pose_valid") is not True:
        raise ValueError("Height pose must be valid")
    error = float(state["height_m"]) - float(state["h_goal_m"])
    if not np.isclose(error, state.get("error_m"), atol=1e-7, rtol=0):
        raise ValueError("Height error mismatch")
    aux = array(
        [
            state["height_m"],
            state["h_goal_m"],
            error,
            state["elapsed_s"],
            state["effort_nm"] if force_valid else 0.0,
            float(force_valid),
            state["confirmation_s"],
            float(state["eligible"]),
        ],
        (8,),
        "arm state",
    )
    if aux[3] < 0 or aux[6] < 0:
        raise ValueError("Elapsed/confirmation time cannot be negative")
    phase = np.zeros(len(PHASES), np.float32)
    phase[PHASES.index(state["phase"])] = 1
    scheduler = parts["scheduler"]
    queue = array(scheduler.get("targets"), (50, 14), "scheduled targets")
    valid = np.asarray(scheduler.get("valid_mask"))
    committed = np.asarray(scheduler.get("committed_mask"))
    if (
        valid.shape != (50,)
        or committed.shape != (50,)
        or valid.dtype != np.bool_
        or committed.dtype != np.bool_
        or np.any(committed & ~valid)
    ):
        raise ValueError("Invalid scheduling masks")
    queue = np.where(valid[:, None], queue, 0)
    result = np.concatenate(
        (
            visual,
            feedback,
            reference.ravel(),
            aux,
            phase,
            queue.ravel(),
            valid.astype(np.float32),
            committed.astype(np.float32),
        )
    )
    if result.shape != (feature_dim + NON_VISUAL_DIM,):
        raise ValueError("State schema dimension mismatch")
    return result
