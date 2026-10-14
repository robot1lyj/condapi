"""Exploration around a preserved physical actor center; NumPy-only contracts."""

import copy

import numpy as np

from yam_pld_rl.block_contract import array
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import feature_dim
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.block_contract import window_context


def validate_collector(manifest):
    profile = manifest.get("collector")
    if profile is None:
        if "actor_contract" in manifest:
            raise ValueError("actor coordinates require an explicit collector")
        return
    smooth = profile.get("kind") == "coherent_trajectory_v1"
    if profile.get("kind") not in {"physical_center_tanh_v1", "coherent_trajectory_v1"}:
        raise ValueError("unknown collector")
    target, actor = manifest["contract"], manifest["actor_contract"]
    validate_contract(target)
    validate_contract(actor)
    if target["gate"].get("version") != 3 or target.get("action_mapping") != ("signed_trajectory_v1" if smooth else "signed_feasible_v1"):
        raise ValueError("collector requires the drained signed action contract")
    for key in set(target) | set(actor):
        allowed = {"radius", "max_residual_step", "status"} | ({"trajectory", "block_steps", "action_mapping"} if smooth else set())
        if key not in allowed and target.get(key) != actor.get(key):
            raise ValueError("collector cannot change gate, time or hardware limits")
    for key in ("radius", "max_residual_step"):
        if np.any(np.asarray(target[key]) < np.asarray(actor[key])):
            raise ValueError("collector must contain the parent action support")
    std = array(profile["latent_std"], (7,), "collector std")
    if np.any(std <= 0) or np.any(std > 1):
        raise ValueError("invalid latent standard deviation")
    if smooth:
        evaluation = manifest["eval_contract"]
        validate_contract(evaluation)
        if evaluation["radius"] != actor["radius"]:
            raise ValueError("eval must preserve the parent target scale")
        for key in set(target) | set(evaluation):
            if key not in {"radius", "max_residual_step", "status"} and target.get(key) != evaluation.get(key):
                raise ValueError("eval trajectory and scope must match collect")
    elif profile.get("eval_preserves_parent_physical_policy") is not True:
        raise ValueError("collector must preserve parent eval")
    parent = profile.get("parent_snapshot_sha256", "")
    if len(parent) != 64 or any(c not in "0123456789abcdef" for c in parent):
        raise ValueError("collector requires a pinned parent snapshot")


def execution_contract(manifest, mode):
    # Only collect expands support. Continuous eval uses the same new decoder
    # and scope guards, while probe/shadow preserve the parent path.
    if mode == "eval" and manifest.get("eval_contract"):
        return manifest["eval_contract"]
    return manifest.get("actor_contract", manifest["contract"]) if mode != "collect" else manifest["contract"]


def center_latent(physical_center, radius, positive, negative):
    center = array(physical_center, (7,), "physical actor center")
    cap = array(radius, (7,), "radius") * np.where(center < 0, negative, positive)
    if np.any(np.abs(center) > cap+1e-7):
        raise ValueError("expanded support does not contain actor center")
    u = np.divide(center, cap, out=np.zeros(7, np.float32), where=cap > 0)
    return np.arctanh(np.clip(u, -1+1e-6, 1-1e-6)).astype(np.float32)


def expanded_manifest(parent, *, parent_snapshot):
    if parent.get("collector"):
        raise ValueError("nested collector conversion is not supported")
    if parent.get("phase") != "candidate_requires_AB" or parent.get("provenance", {}).get("actor_updates", 0) <= 0:
        raise ValueError("expanded collector requires a learned parent")
    out = copy.deepcopy(parent)
    out["actor_contract"] = copy.deepcopy(parent["contract"])
    out["contract"]["radius"] = [.08]*6+[.05]+[.08]*6+[.05]
    out["contract"]["max_residual_step"] = [.03]*6+[.02]+[.03]*6+[.02]
    out["contract"]["status"] = "expanded collection around preserved learned physical center; first episode audit required"
    out["collector"] = {"kind": "physical_center_tanh_v1", "latent_std": [.5]*6+[.2],
                        "parent_snapshot_sha256": parent_snapshot,
                        "eval_preserves_parent_physical_policy": True,
                        "density_semantics": "transformed center Gaussian followed by tanh; latent density"}
    out["provenance"] = {**parent["provenance"], "collector_parent_snapshot_sha256": parent_snapshot,
                         "collector_changes_actor_weights": False}
    out["robot_validated"] = False
    validate_collector(out)
    return out


def smooth_manifest(parent, *, parent_snapshot):
    out = expanded_manifest(parent, parent_snapshot=parent_snapshot)
    c = out["contract"]
    c.update(action_mapping="signed_trajectory_v1", block_steps=36,
             max_residual_step=[.015]*6+[.02]+[.015]*6+[.02],
             trajectory={"kind": "prefix_trajectory_v1", "ramp_steps": 12, "noise_tau_s": .5, "bridge_velocity_gain": .25,
                         "max_residual_second_difference": [.006]*6+[.01]+[.006]*6+[.01],
                         "max_target_second_difference": [.02]*6+[.10]+[.02]*6+[.10],
                         "forward_x_max_m": {"left": .40, "right": .40}, "approach_z_margin_m": .12},
             status="continuous collect candidate; target radius .08; trajectory and workspace verification required")
    out["eval_contract"] = copy.deepcopy(c)
    for key in ("radius", "max_residual_step"):
        out["eval_contract"][key] = copy.deepcopy(parent["contract"][key])
    out["collector"].update(kind="coherent_trajectory_v1", eval_preserves_parent_physical_policy=False,
                            eval_preserves_parent_target_scale=True,
                            density_semantics="conditional Gaussian innovation then tanh; before deterministic trajectory/geometry decoding")
    validate_collector(out)
    return out


def native_trajectory_manifest(manifest, *, behavior_snapshot):
    """Export a newly learned trajectory actor in its OWN action coordinates.

    Actor-1's .02 contract is only for decoding Actor-1 behavior. A new actor
    trained on trajectory targets must never be routed through that old scale.
    """
    out = copy.deepcopy(manifest)
    if not out["contract"].get("trajectory"):
        raise ValueError("native trajectory export requires a trajectory contract")
    out["actor_contract"] = copy.deepcopy(out["contract"])
    out["eval_contract"] = copy.deepcopy(out["contract"])
    out["collector"] = {"kind": "coherent_trajectory_v1", "native_actor_coordinates": True,
                        "parent_snapshot_sha256": behavior_snapshot,
                        "latent_std": [.5]*6+[.2],
                        "eval_preserves_parent_physical_policy": False,
                        "eval_preserves_parent_target_scale": True,
                        "density_semantics": "conditional Gaussian innovation then tanh; before deterministic trajectory/geometry decoding"}
    validate_collector(out)
    return out


def validate_collector_trace(p, manifest):
    """Check physical coordinates and parent observation; no network execution."""
    if not manifest.get("collector"):
        return
    smooth = manifest["collector"]["kind"] == "coherent_trajectory_v1"
    c = execution_contract(manifest, p["mode"])
    if p["mode"] not in ({"collect", "eval"} if smooth else {"collect"}) or p.get("execution_contract_sha256") != fingerprint(c):
        raise ValueError("expanded replay requires collect execution coordinates")
    ac = manifest["actor_contract"]
    base, arm, d = p.get("physical_base", p["base"]), p["arm"], p["delay_steps"]
    ap, an = feasible_bounds(base, p["previous_command"], arm, d, ac,
                             context=p.get("trajectory_context") if ac.get("trajectory") else None)
    if p["gate"].get("apply_residual") is False:
        ap, an = np.zeros(7, np.float32), np.zeros(7, np.float32)
    ax = observation_vector(np.asarray(p["x"])[:1536], p["state"], base, p["previous_command"],
                            arm, d, p["elapsed_seconds"], ap, manifest["normalization"], negative_scale=an,
                            context=window_context(p["gate"], p["start_tick"], p["base_z_m"][arm], ac["action_dt_s"]))
    mean_u = array(p["actor_mean_u"], (7,), "parent mean action")
    if np.any(np.abs(mean_u) > 1):
        raise ValueError("invalid parent action")
    offset = 0 if arm == "left" else 7
    center = map_action(mean_u, ap, an) * np.asarray(ac["radius"][offset:offset+7], np.float32)
    if smooth:
        radius = np.asarray(c["radius"][offset:offset+7])
        center = np.clip(center, -radius*np.asarray(p["negative_scale"]), radius*np.asarray(p["scale"]))
    latent = center_latent(center, c["radius"][offset:offset+7], p["scale"], p["negative_scale"])
    for key, value, shape in (("actor_context", ax[2264:], (feature_dim(ac)-2264,)), ("actor_scale", ap, (7,)),
                              ("actor_negative_scale", an, (7,)), ("actor_physical_center", center, (7,)),
                              ("collector_latent_center", latent, (7,)),
                              ("collector_latent_std", manifest["collector"]["latent_std"], (7,))):
        if not np.allclose(array(p[key], shape, key), value, atol=1e-6, rtol=0):
            raise ValueError(f"collector provenance mismatch: {key}")
    if smooth and p["gate"].get("apply_residual"):
        noise = p["collector_noise"]
        expected = latent + np.asarray(manifest["collector"]["latent_std"])*np.asarray(noise["value"])
        if not np.allclose(np.tanh(expected), p["sample_u"], atol=1e-6, rtol=0):
            raise ValueError("correlated sample mismatch")
        rho = noise["rho"]
        if not 0 <= rho < 1 or not np.allclose(noise["value"], rho*np.asarray(noise["previous"])+np.sqrt(1-rho*rho)*np.asarray(noise["innovation"]), atol=1e-6, rtol=0):
            raise ValueError("correlated noise transition mismatch")
