"""Synthetic contract fixtures only; never robot data or physical limits."""

import numpy as np
from yam_pld_rl.block_contract import FEATURE_DIM
from yam_pld_rl.block_contract import SCHEMA
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_scale
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.intervention import V2_GATE
from yam_pld_rl.kinematics import MODEL_SHA256


def contract():
    return {"schema": SCHEMA, "gate": V2_GATE.copy(), "fk_model_sha256": MODEL_SHA256, "joint_unit": "radian", "gripper_unit": "0_closed_1_open",
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
        n = delay + c["block_steps"] + attempt % 3
        scale = feasible_scale(base, previous, arm, delay, c)
        for block in range(3):
            u = rng.uniform(-0.5, 0.5, 7).astype(np.float32)
            a = scale * u if mode == "collect" else np.zeros(7, dtype=np.float32)
            x = observation_vector(rng.normal(size=1536), state, base, previous, arm, delay, block * n / 30, scale, stats())
            assert len(x) == FEATURE_DIM
            action = decode(base, a, arm, delay, c)
            trace = {**ids, "decision_id": f"d{block}", "arm": arm, "mode": mode, "epoch": 0, "window_id": "e0:w1",
                     "observation_id": block, "request_id": block, "gate": {"active": True},
                     "schema": c["schema"], "snapshot_sha256": snapshot, "contract_sha256": fingerprint(c),
                     "start_tick": block * n, "end_tick": (block + 1) * n, "delay_steps": delay,
                     "elapsed_seconds": block * n / 30, "base": base.tolist(), "state": state.tolist(),
                     "previous_command": previous.tolist(), "scale": scale.tolist(), "sample_u": u.tolist(),
                     "action": a.tolist(), "x": x.tolist(), "actions": action.tolist()}
            receipt = {"decision_id": f"d{block}", "status": "executed", "source_verified": True, "start_tick": block * n,
                       "end_tick": (block + 1) * n, "epoch": 0, "ticks": list(range(block * n, (block + 1) * n)),
                       "monotonic_times": (10_000. + np.arange(block * n, (block + 1) * n + 1) / 30).tolist(),
                       "submitted_actions": action[:n].tolist(), "client_modified": False, "human_intervention": False,
                       "observation_ref": f"fixture/ep{attempt}/obs{block}", "next_observation_ref": f"fixture/ep{attempt}/obs{block + 1}"}
            result.append({"proposal": trace, "receipt": receipt})
        labels.append({**ids, "status": "held" if attempt % 2 else "empty", "reviewer": "synthetic-oracle",
                       "evidence": ["synthetic, not visual evaluation"], "lift_m": 0.1, "view_valid": True,
                       "held_after_lift": attempt % 2 == 1, "final_decision_id": "d2", "outcome_tick": 3 * n - 1, "terminal_tick": 3 * n, "close_tick": n,
                       "split": "val" if attempt == attempts - 1 else "train"})
    return result, labels


def inference_case(runtime, *, arm="left", mode="collect", epoch=0):
    """Current open state and a causal Pi closure forecast; no client arm flag."""
    rng = np.random.default_rng(1431 + epoch)
    state = np.full(14, .123456789)
    state[[6, 13]] = .9
    obs = {"prompt": runtime.contract["task"], "observation.state": state}
    obs.update({f"observation.images.{view}_rgb": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
                for view in ("top", "left", "right")})
    base = np.tile(state.astype(np.float64), (50, 1))
    base[:, :6] = .12345678901234
    base[:, 7:13] = .12345678901234
    base[:, 6 if arm == "left" else 13] = .9 - .06 * np.minimum(np.maximum(np.arange(50)-3, 0), 9)
    rtc = {"delay_steps": 3, "observation_policy_tick": 0}
    request = {**runtime.metadata, "mode": mode, "run_id": "fixture", "episode_id": f"ep{epoch}",
               "epoch": epoch, "request_id": 1, "observation_id": 1, "previous_command": base[0]}
    return obs, rtc, request, {"actions": base}
