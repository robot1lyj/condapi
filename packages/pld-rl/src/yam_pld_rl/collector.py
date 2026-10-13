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
    if profile.get("kind") != "physical_center_tanh_v1":
        raise ValueError("unknown collector")
    target, actor = manifest["contract"], manifest["actor_contract"]
    validate_contract(target)
    validate_contract(actor)
    if target["gate"].get("version") != 3 or target.get("action_mapping") != "signed_feasible_v1":
        raise ValueError("collector requires the drained signed action contract")
    for key in set(target) | set(actor):
        if key not in {"radius", "max_residual_step", "status"} and target.get(key) != actor.get(key):
            raise ValueError("collector cannot change gate, time or hardware limits")
    for key in ("radius", "max_residual_step"):
        if np.any(np.asarray(target[key]) < np.asarray(actor[key])):
            raise ValueError("collector must contain the parent action support")
    std = array(profile["latent_std"], (7,), "collector std")
    if np.any(std <= 0) or np.any(std > 1):
        raise ValueError("invalid latent standard deviation")
    if profile.get("eval_preserves_parent_physical_policy") is not True:
        raise ValueError("collector must preserve parent eval")
    parent = profile.get("parent_snapshot_sha256", "")
    if len(parent) != 64 or any(c not in "0123456789abcdef" for c in parent):
        raise ValueError("collector requires a pinned parent snapshot")


def execution_contract(manifest, mode):
    # Eval/probe/shadow preserve the parent path. Only collect expands support.
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


def validate_collector_trace(p, manifest):
    """Check physical coordinates and parent observation; no network execution."""
    if not manifest.get("collector"):
        return
    if p["mode"] != "collect" or p.get("execution_contract_sha256") != fingerprint(manifest["contract"]):
        raise ValueError("expanded replay requires collect execution coordinates")
    ac, c = manifest["actor_contract"], manifest["contract"]
    base, arm, d = p.get("physical_base", p["base"]), p["arm"], p["delay_steps"]
    ap, an = feasible_bounds(base, p["previous_command"], arm, d, ac)
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
    latent = center_latent(center, c["radius"][offset:offset+7], p["scale"], p["negative_scale"])
    for key, value, shape in (("actor_context", ax[2264:], (feature_dim(ac)-2264,)), ("actor_scale", ap, (7,)),
                              ("actor_negative_scale", an, (7,)), ("actor_physical_center", center, (7,)),
                              ("collector_latent_center", latent, (7,)),
                              ("collector_latent_std", manifest["collector"]["latent_std"], (7,))):
        if not np.allclose(array(p[key], shape, key), value, atol=1e-6, rtol=0):
            raise ValueError(f"collector provenance mismatch: {key}")
