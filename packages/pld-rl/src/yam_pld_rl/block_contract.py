"""Causal PLD action-block contract; NumPy only, no robot or learner imports."""

import hashlib
import json

import numpy as np

from yam_pld_rl.data import action_coordinates
from yam_pld_rl.kinematics import MODEL_SHA256

SCHEMA = "pld_sac_block_v2"
HORIZON = 50
FEATURE_DIM = 1536 + 14 + 50 * 14 + 14 + 2 + 2 + 7


def feature_dim(contract):
    return FEATURE_DIM + (12 if contract["gate"].get("version") == 3 else 0)


def window_context(gate, tick, height, dt):
    closed = gate.get("close_tick") is not None
    return np.asarray([gate.get("apply_residual") is True, closed,
                       height-gate["close_z"] if closed else 0.,
                       (tick-gate["close_tick"])*dt if closed else 0.,
                       max(0, gate.get("drain_until_tick", tick)-tick)*dt], dtype=np.float32)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array(value, shape, name):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite {shape}")
    return a


def integer(value, name, lower=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError(f"{name}: invalid integer")
    return value


def validate_contract(c):
    if c["schema"] != SCHEMA or c["joint_unit"] != "radian" or c["gripper_unit"] != "0_closed_1_open":
        raise ValueError("unsupported schema or units")
    if not c.get("limits_evidence") or not c.get("base_weights_sha256") or not c.get("task"):
        raise ValueError("physical limits evidence, task and base identity required")
    from yam_pld_rl.intervention import DEFAULT_GATE  # noqa: PLC0415 -- gate imports the array contract
    from yam_pld_rl.intervention import LEGACY_GATE  # noqa: PLC0415 -- gate imports the array contract
    from yam_pld_rl.intervention import V2_GATE  # noqa: PLC0415

    if c.get("gate") not in (DEFAULT_GATE, V2_GATE, LEGACY_GATE) or c.get("fk_model_sha256") != MODEL_SHA256:
        raise ValueError("explicit pinned Thor gate and FK geometry required")
    if c.get("action_mapping") not in (None, "signed_feasible_v1", "signed_trajectory_v1"):
        raise ValueError("unsupported action mapping")
    if c["gate"].get("version") == 3 and c.get("reward_contract") != "drained_grasp_v1":
        raise ValueError("gate v3 requires drained outcome contract")
    if c.get("baseline_transform") not in (None, "sdk_gripper_endpoints_v1"):
        raise ValueError("unsupported physical baseline transform")
    k = integer(c["block_steps"], "block_steps", 4)
    trajectory = c.get("action_mapping") == "signed_trajectory_v1"
    if k > (36 if trajectory else 20) or c["action_dt_s"] != 1 / 30 or not 0 < c["discount_per_second"] <= 1:
        raise ValueError("invalid block/time contract")
    lo, hi = array(c["lower"], (14,), "lower"), array(c["upper"], (14,), "upper")
    radius = array(c["radius"], (14,), "radius")
    step = array(c["max_target_step"], (14,), "max_target_step")
    if np.any(lo >= hi) or np.any(radius <= 0) or np.any(step <= 0):
        raise ValueError("invalid limits/radius/step")
    directional = ("positive_radius" in c, "negative_radius" in c)
    if any(directional):
        if not trajectory or not all(directional):
            raise ValueError("directional budgets require both caps and the trajectory decoder")
        for key in ("positive_radius", "negative_radius"):
            cap = array(c[key], (14,), key)
            if np.any(cap <= 0) or np.any(cap > radius+1e-7):
                raise ValueError("directional cap outside coordinate radius")
    if "max_residual_step" in c and np.any(array(c["max_residual_step"], (14,), "max_residual_step") <= 0):
        raise ValueError("invalid residual increment guard")
    if not 0 < c["max_attempt_seconds"] <= 30:
        raise ValueError("explicit bounded attempt duration required")
    if trajectory:
        t = c["trajectory"]
        if "short_closing_min_steps" in t and not 5 <= integer(t["short_closing_min_steps"], "short closing minimum", 5) <= 8:
            raise ValueError("short closing requires 5 to 8 editable ticks")
        if (t["kind"] != "prefix_trajectory_v1" or not 3 <= t["ramp_steps"] <= 14
                or not 0 < t["noise_tau_s"] <= 1 or not 0 <= t["approach_z_margin_m"] <= .2
                or not 0 <= t["bridge_velocity_gain"] <= 1
                or np.any(array(t["max_residual_second_difference"], (14,), "residual acceleration") <= 0)
                or np.any(array(t["max_target_second_difference"], (14,), "target acceleration") <= 0)
                or not all(0 < t["forward_x_max_m"][a] < 1 for a in ("left", "right"))):
            raise ValueError("invalid prefix trajectory contract")
    if c.get("grasp_z_range_m"):
        if not c.get("grasp_scope_evidence"):
            raise ValueError("scene scope calibration evidence required")
        for arm in ("left", "right"):
            limits = array(c["grasp_z_range_m"][arm], (2,), "grasp height scope")
            if limits[0] >= limits[1]:
                raise ValueError("invalid grasp height scope")


def envelope_weights(delay, c):
    k = c["block_steps"]
    if delay < 0 or delay + k >= HORIZON:
        raise ValueError("editable block must leave an unchanged suffix")
    w = np.zeros(HORIZON, dtype=np.float32)
    # Zero at both ends prevents residual discontinuities at Pi handoff.
    w[delay : delay + k] = np.minimum(np.arange(k), np.arange(k)[::-1]) / ((k - 1) // 2)
    return w


def physical_baseline(raw, delay, c):
    """Physical commands after SDK gripper endpoints; raw RTC stays separate."""
    array(raw, (50, 14), "raw Pi")
    out = np.asarray(raw).copy()
    if c.get("baseline_transform") != "sdk_gripper_endpoints_v1":
        return out
    lo, hi = (np.asarray(c[k]) for k in ("lower", "upper"))
    joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    if np.any(out[:, joints] < lo[joints]) or np.any(out[:, joints] > hi[joints]):
        raise ValueError("Pi joint target outside declared SDK bounds")
    for index in (6, 13):
        out[:, index] = np.clip(out[:, index], lo[index], hi[index])
    return out


def wire_plan(physical, raw, delay):
    """Preserve native committed prefix bytes, including existing SDK mapping."""
    out = np.asarray(physical).copy()
    out[:delay] = np.asarray(raw)[:delay]
    return out


def feasible_scale(base, previous_command, arm, delay, c):
    """Symmetric feasible interval; no clipping which would change zero/Pi.

    Includes the unchanged next Pi tick in the continuity check. The previously
    committed prefix is authoritative and never projected. These are joint
    command guards, not collision or force guarantees.
    """
    base = array(base, (50, 14), "base")
    previous_command = array(previous_command, (14,), "previous_command")
    if arm not in ("left", "right"):
        raise ValueError("one active arm required")
    w = envelope_weights(delay, c)
    lo, hi, step = (np.asarray(c[key], dtype=np.float32) for key in ("lower", "upper", "max_target_step"))
    end = delay + c["block_steps"]
    b = base[delay : end + 1]
    anchor = base[delay - 1] if delay else previous_command
    differences = np.diff(np.vstack((anchor, b)), axis=0)
    if np.any(b < lo) or np.any(b > hi) or np.any(np.abs(differences) > step + 1e-7):
        raise ValueError("base block already violates limits; PLD cannot silently repair it")
    axes = slice(0, 7) if arm == "left" else slice(7, 14)
    bound = np.asarray(c["radius"], dtype=np.float32)[axes].copy()
    for i in range(len(b)):
        wi = w[delay + i]
        if wi > 0:
            bound = np.minimum(bound, np.minimum(b[i, axes] - lo[axes], hi[axes] - b[i, axes]) / wi)
        dw = abs(wi - (w[delay + i - 1] if delay + i else 0))
        if dw > 0:
            bound = np.minimum(bound, (step[axes] - np.abs(differences[i, axes])) / dw)
            if "max_residual_step" in c:
                bound = np.minimum(bound, np.asarray(c["max_residual_step"])[axes] / dw)
    return np.clip(bound / np.asarray(c["radius"])[axes], 0, 1).astype(np.float32)


def feasible_bounds(base, previous_command, arm, delay, c, *, context=None):
    """Positive/negative magnitudes; zero always means the physical Pi plan."""
    if c.get("action_mapping") == "signed_trajectory_v1":
        from yam_pld_rl.trajectory import feasible_trajectory_bounds  # noqa: PLC0415
        if context is None:
            raise ValueError("trajectory planning context required")
        return feasible_trajectory_bounds(base, previous_command, arm, c, context)
    symmetric = feasible_scale(base, previous_command, arm, delay, c)
    if c.get("action_mapping") != "signed_feasible_v1":
        return symmetric, symmetric.copy()
    base = array(base, (50, 14), "base")
    axes = slice(0, 7) if arm == "left" else slice(7, 14)
    radius = np.asarray(c["radius"], dtype=np.float32)[axes]
    positive, negative = radius.copy(), radius.copy()
    lo, hi, step = (np.asarray(c[k], dtype=np.float32)[axes] for k in ("lower", "upper", "max_target_step"))
    w = envelope_weights(delay, c)
    end = delay + c["block_steps"]
    anchor = base[delay-1] if delay else array(previous_command, (14,), "previous")
    differences = np.diff(np.vstack((anchor, base[delay:end+1])), axis=0)[:, axes]
    for i, row in enumerate(range(delay, end+1)):
        wi = w[row]
        if wi > 0:
            positive = np.minimum(positive, (hi-base[row, axes])/wi)
            negative = np.minimum(negative, (base[row, axes]-lo)/wi)
        dw = wi - (w[row-1] if row else 0)
        if dw:
            signed_diff = np.sign(dw)*differences[i]
            positive = np.minimum(positive, (step-signed_diff)/abs(dw))
            negative = np.minimum(negative, (step+signed_diff)/abs(dw))
            if "max_residual_step" in c:
                cap = np.asarray(c["max_residual_step"])[axes]/abs(dw)
                positive, negative = np.minimum(positive, cap), np.minimum(negative, cap)
    return tuple(np.clip(v/radius, 0, 1).astype(np.float32) for v in (positive, negative))


def map_action(u, positive, negative):
    u = array(u, (7,), "latent action")
    return u * np.where(u < 0, negative, positive)


def decode(base, scaled_action, arm, delay, c, *, context=None):
    a = array(scaled_action, (7,), "scaled_action")
    if np.any(np.abs(a) > 1.000001):
        raise ValueError("residual outside support")
    array(base, (50, 14), "base")
    if c.get("action_mapping") == "signed_trajectory_v1":
        from yam_pld_rl.trajectory import trajectory_decode  # noqa: PLC0415
        if context is None:
            raise ValueError("trajectory planning context required")
        return trajectory_decode(base, a, arm, c, context)[0]
    out = np.asarray(base).copy()
    if out.dtype.kind != "f":
        raise ValueError("physical actions must have floating dtype")
    axes = slice(0, 7) if arm == "left" else slice(7, 14)
    offset = a * np.asarray(c["radius"], dtype=np.float32)[axes]
    weights = envelope_weights(delay, c)
    # Do not even round-trip the committed prefix / inactive arm / zero head.
    for row in np.flatnonzero(weights):
        out[row, axes] += weights[row] * offset
    if "max_residual_step" in c and np.any(np.abs(np.diff(out - base, axis=0)) > np.asarray(c["max_residual_step"]) + 2e-7):
        raise ValueError("residual increment exceeds the Thor guard")
    return out


def observation_vector(features, state, base, previous_command, arm, delay, elapsed, scale, stats,
                       *, negative_scale=None, context=None):
    f = array(features, (1536,), "visual features")
    state = array(state, (14,), "state")
    base = array(base, (50, 14), "base")
    previous_command = array(previous_command, (14,), "previous_command")
    sm, ss = np.asarray(stats["state_mean"]), np.asarray(stats["state_std"])
    am, ast = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])
    if not np.isfinite(elapsed) or elapsed < 0 or np.any(ss <= 0) or np.any(ast <= 0):
        raise ValueError("invalid normalization or causal elapsed time")
    extra = []
    if negative_scale is not None or context is not None:
        extra = [*array(negative_scale, (7,), "negative scale"), *array(context, (5,), "window context")]
    x = np.concatenate((
        f, (state - sm) / ss, ((action_coordinates(base, state) - am) / ast).ravel(),
        (previous_command - sm) / ss, [arm == "left", arm == "right", delay / 50, elapsed / 30], extra, scale,
    )).astype(np.float32)
    return array(x, (FEATURE_DIM+len(extra),), "observation vector")
