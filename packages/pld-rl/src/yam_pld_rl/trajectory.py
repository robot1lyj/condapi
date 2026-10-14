"""Finite, prefix-anchored residual trajectories and stationary correlated exploration.

No robot I/O. Decoder state is reconstructed from the recorded Pi plan, state,
previous command and gate; exploration history only belongs to the behavior sampler.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from yam_pld_rl.block_contract import array
from yam_pld_rl.kinematics import YamKinematics

KIND = "prefix_trajectory_v1"


@lru_cache(maxsize=1)
def model_fk():
    return YamKinematics(Path(__file__).resolve().parents[4]/"third_party/yam-fk-model/yam_linear4310.xml")


def in_workspace(xyz, arm, c):
    xyz = np.asarray(xyz)
    cfg = c["trajectory"]
    low, high = c["grasp_z_range_m"][arm]
    return ((xyz[..., 0] <= cfg["forward_x_max_m"][arm])
            & (xyz[..., 2] >= low)
            & (xyz[..., 2] <= high+cfg["approach_z_margin_m"]))


def planning_context(base, previous, state, gate, arm, delay, c):
    b = np.asarray(base, np.float64)
    offset = 0 if arm == "left" else 7
    xyz = model_fk().positions(b[:, offset:offset+6])
    measured = model_fk().position(np.asarray(state)[offset:offset+6])
    end = min(50, delay+c["block_steps"])
    flags = in_workspace(xyz, arm, c)
    # Only the next closure/release cycle is editable. Future lift or opening
    # ends the plan BEFORE transport/placement, not after a measured callback.
    g = b[:, offset+6]
    closed = gate.get("close_tick") is not None
    hits = np.flatnonzero(float(state[offset+6])-g[delay:] >= c["gate"]["close_drop"])
    close = delay if closed else (delay+int(hits[0]) if len(hits) else None)
    lift_ceiling = None
    if close is not None:
        ref = gate["close_z"] if closed else min(float(measured[2]), float(xyz[delay:close+1, 2].min()))
        flags[close:] &= xyz[close:, 2] < ref+c["gate"]["lift_m"]
        lift_ceiling = float(ref+c["gate"]["lift_m"])
        minimum = np.minimum.accumulate(g[close:])
        flags[close:] &= g[close:]-minimum < c["gate"]["reopen_delta"]
    bad = np.flatnonzero(~flags[delay:end])
    if len(bad):
        end = delay+int(bad[0])
    bias, weight = np.zeros((50, 7)), np.zeros(50)
    reason = None
    if gate.get("apply_residual") is False:
        reason = "draining"
    elif not in_workspace(measured, arm, c):
        reason = "current_pose_outside_grasp_workspace"
    elif end-delay < 9:
        reason = "insufficient_grasp_only_horizon"
    if reason is None:
        n = end-delay
        ramp = min(c["trajectory"]["ramp_steps"], max(3, (n-1)//2))
        fade = n-1-ramp
        q = b[:, offset:offset+7]
        if delay >= 3:
            velocity = q[delay-1]-q[delay-2]
            start = q[delay-1]+velocity-q[delay]
            # Preserve the first physical step exactly, but damp extrapolated
            # velocity disagreement so re-planning does not ring around a target.
            derivative = c["trajectory"]["bridge_velocity_gain"]*(velocity-(q[delay+2]-q[delay])/2)
        else:
            # With too little committed history, start at the Pi plan. Real
            # production uses delay=10; do not fabricate an unobserved velocity.
            start, derivative = np.zeros(7), np.zeros(7)
        u = np.arange(ramp+1)/ramp
        s = 10*u**3-15*u**4+6*u**5
        h = u-6*u**3+8*u**4-3*u**5
        bias[delay:delay+ramp+1] = (1-s[:, None])*start+h[:, None]*ramp*derivative
        weight[delay:delay+ramp+1] = s
        u = np.arange(fade+1)/fade
        weight[delay+ramp:end] = 1-(10*u**3-15*u**4+6*u**5)
        weight = np.clip(weight, 0, 1)
    context = {"kind": KIND, "delay_steps": delay, "end_index": end if reason is None else delay, "reason": reason,
               "lift_guard_start_index": close, "lift_ceiling_m": lift_ceiling,
               "bias": bias.tolist(), "weight": weight.tolist()}
    if reason is None and not _bounds(b, previous, arm, c, context)[2]:
        context.update(reason="baseline_bridge_exceeds_motion_guards", bias=np.zeros((50, 7)).tolist(),
                       weight=np.zeros(50).tolist(), end_index=delay)
    return context


def _bounds(base, previous, arm, c, context):
    off = 0 if arm == "left" else 7
    b = np.asarray(base, float)[:, off:off+7]
    bias, w = np.asarray(context["bias"]), np.asarray(context["weight"])
    radius = np.asarray(c["radius"])[off:off+7]
    coeff = w[:, None]*radius
    low, high = np.full(7, -1.), np.ones(7)
    ok = True
    def restrict(a, coef, lo, hi):
        nonlocal low, high, ok
        nz = np.abs(coef)>1e-12
        ok = ok and bool(np.all(nz | ((a >= lo-1e-7)&(a <= hi+1e-7))))
        denom = np.where(nz, coef, 1.)
        lower, upper = (lo-a)/denom, (hi-a)/denom
        low = np.maximum(low, np.where(nz, np.minimum(lower, upper), -np.inf).max(0))
        high = np.minimum(high, np.where(nz, np.maximum(lower, upper), np.inf).min(0))
    restrict(bias, coeff, -radius, radius)
    restrict(b+bias, coeff, np.asarray(c["lower"])[off:off+7], np.asarray(c["upper"])[off:off+7])
    velocity = np.asarray(c["max_residual_step"])[off:off+7]
    acceleration = np.asarray(c["trajectory"]["max_residual_second_difference"])[off:off+7]
    # At the first new tick the Pi reference itself has changed. The anchor
    # cancels that reference jump; constrain the absolute commands there,
    # rather than rejecting a physically continuous bridge for its delta.
    interior_velocity = np.arange(49) >= context["delay_steps"]
    restrict(np.diff(bias, axis=0)[interior_velocity], np.diff(coeff, axis=0)[interior_velocity], -velocity, velocity)
    interior = np.arange(48) > context["delay_steps"]
    restrict(np.diff(bias, n=2, axis=0)[interior], np.diff(coeff, n=2, axis=0)[interior], -acceleration, acceleration)
    step = np.asarray(c["max_target_step"])[off:off+7]
    q = np.vstack((np.asarray(previous)[off:off+7], b+bias))
    dc = np.diff(np.vstack((np.zeros(7), coeff)), axis=0)
    # Existing committed prefix is immutable; only the suffix and tail are
    # constrained here. The initial rows can belong to a different proposal.
    editable = (np.abs(bias).sum(1)+np.abs(coeff).sum(1))>1e-12
    check = editable | np.r_[False, editable[:-1]]
    if check.any():
        restrict(np.diff(q, axis=0)[check], dc[check], -step, step)
    check_acc = editable[2:] | editable[1:-1] | editable[:-2]
    if check_acc.any():
        acceleration = np.asarray(c["trajectory"]["max_target_second_difference"])[off:off+7]
        restrict(np.diff(b+bias, n=2, axis=0)[check_acc], np.diff(coeff, n=2, axis=0)[check_acc], -acceleration, acceleration)
    ok = ok and bool(np.all(low<=1e-7) and np.all(high>=-1e-7))
    return np.clip(high, 0, 1).astype(np.float32), np.clip(-low, 0, 1).astype(np.float32), ok


def feasible_trajectory_bounds(base, previous, arm, c, context):
    if context["reason"] is not None:
        return np.zeros(7, np.float32), np.zeros(7, np.float32)
    positive, negative, ok = _bounds(base, previous, arm, c, context)
    if not ok:
        raise ValueError("trajectory support excludes zero")
    return positive, negative


def trajectory_decode(base, action, arm, c, context):
    """Deterministic environment map from desired residual to guarded targets.

    The critic action is the proposed normalized target; any deterministic
    geometry projection is part of this decoder and is reproduced in replay.
    """
    b = np.asarray(base)
    a = array(action, (7,), "trajectory target")
    if context["reason"] is not None:
        return b.copy(), {"factor": 0., "reason": context["reason"]}
    off = 0 if arm == "left" else 7
    bias, w = np.asarray(context["bias"]), np.asarray(context["weight"])
    offset = a*np.asarray(c["radius"])[off:off+7]
    affected = (np.abs(bias).sum(1)+np.abs(w))>1e-12
    for factor in (1., .75, .5, .25, 0.):
        residual = bias+w[:, None]*offset*factor
        changed = b.copy()
        changed[:, off:off+7] += residual
        xyz = model_fk().positions(changed[affected, off:off+6])
        allowed = in_workspace(xyz, arm, c)
        if context["lift_ceiling_m"] is not None:
            closed_rows = np.flatnonzero(affected) >= context["lift_guard_start_index"]
            allowed &= ~closed_rows | (xyz[:, 2] < context["lift_ceiling_m"])
        if np.all(allowed):
            return changed, {"factor": factor, "reason": "inside_grasp_workspace" if factor==1 else "geometry_projection"}
    return b.copy(), {"factor": 0., "reason": "bridge_outside_grasp_workspace"}


class CorrelatedNoise:
    def __init__(self):
        self.key, self.tick, self.value = None, None, np.zeros(7, np.float32)

    def advance(self, key, tick, dt, tau, innovation):
        previous = self.value.copy() if key==self.key else np.zeros(7, np.float32)
        rho = float(np.exp(-(tick-self.tick)*dt/tau)) if key==self.key else 0.
        if not 0<=rho<1:
            raise ValueError("nonmonotonic exploration clock")
        value = rho*previous+np.sqrt(1-rho*rho)*array(innovation, (7,), "noise innovation")
        self.key, self.tick, self.value = key, tick, value.astype(np.float32)
        return {"rho": rho, "previous": previous, "innovation": innovation, "value": self.value.copy()}
