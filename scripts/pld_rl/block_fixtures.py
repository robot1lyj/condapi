"""Synthetic contract fixtures only; never robot data or physical limits."""

import numpy as np
from yam_pld_rl.block_contract import FEATURE_DIM
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_scale
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import observation_vector


def contract():
    return {"schema": "pld_sac_block_v1", "joint_unit": "radian", "gripper_unit": "0_closed_1_open",
            "task": "synthetic fixture; not a robot task", "base_weights_sha256": "fixture-only",
            "limits_evidence": "synthetic; NEVER use on hardware", "lower": [-2.] * 14, "upper": [2.] * 14,
            "radius": [0.01] * 14, "max_target_step": [0.2] * 14, "block_steps": 8,
            "action_dt_s": 1 / 30, "discount_per_second": 0.99, "max_attempt_seconds": 15}


def stats():
    return {"state_mean": [0.] * 14, "state_std": [1.] * 14, "action_mean": [0.] * 14, "action_std": [1.] * 14}


def records(c, snapshot, *, attempts=8):
    result, labels = [], []
    rng = np.random.default_rng(781)
    for attempt in range(attempts):
        arm, mode = ("left" if attempt % 2 else "right"), ("probe" if attempt % 3 == 0 else "collect")
        ids = {"run_id": "synthetic", "episode_id": f"ep{attempt}", "attempt_id": f"a{attempt}"}
        base = np.full((50, 14), 0.12345678901234, dtype=np.float64)
        state, previous, delay = base[0].copy(), base[0].copy(), attempt % 3
        n = delay + c["block_steps"]
        scale = feasible_scale(base, previous, arm, delay, c)
        for block in range(3):
            u = rng.uniform(-0.5, 0.5, 7).astype(np.float32)
            a = scale * u if mode == "collect" else np.zeros(7, dtype=np.float32)
            x = observation_vector(rng.normal(size=1536), state, base, previous, arm, delay, block * n / 30, scale, stats())
            assert len(x) == FEATURE_DIM
            action = decode(base, a, arm, delay, c)
            trace = {**ids, "decision_id": f"d{block}", "arm": arm, "mode": mode,
                     "schema": c["schema"], "snapshot_sha256": snapshot, "contract_sha256": fingerprint(c),
                     "start_tick": block * n, "end_tick": (block + 1) * n, "delay_steps": delay,
                     "elapsed_seconds": block * n / 30, "base": base.tolist(), "state": state.tolist(),
                     "previous_command": previous.tolist(), "scale": scale.tolist(), "sample_u": u.tolist(),
                     "action": a.tolist(), "x": x.tolist(), "actions": action.tolist()}
            receipt = {"decision_id": f"d{block}", "status": "executed", "start_tick": block * n,
                       "end_tick": (block + 1) * n, "epoch": 0, "ticks": list(range(block * n, (block + 1) * n)),
                       "monotonic_times": (10_000. + np.arange(block * n, (block + 1) * n + 1) / 30).tolist(),
                       "submitted_actions": action[:n].tolist(), "client_modified": False, "human_intervention": False,
                       "observation_ref": f"fixture/ep{attempt}/obs{block}", "next_observation_ref": f"fixture/ep{attempt}/obs{block + 1}"}
            result.append({"proposal": trace, "receipt": receipt})
        labels.append({**ids, "status": "held" if attempt % 2 else "empty", "reviewer": "synthetic-oracle",
                       "evidence": ["synthetic, not visual evaluation"], "lift_m": 0.1, "view_valid": True,
                       "held_after_lift": attempt % 2 == 1, "final_decision_id": "d2", "outcome_tick": 3 * n,
                       "split": "val" if attempt == attempts - 1 else "train"})
    return result, labels
