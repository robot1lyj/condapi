"""NumPy-only integration checks: native recording to reviewed SAC transitions."""

# ruff: noqa: E402, PT009, PT027

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from block_fixtures import contract
from block_fixtures import records
from block_fixtures import stats
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_scale
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_replay import compile_replay
from yam_pld_rl.curator import associate_rewards
from yam_pld_rl.curator import physical_candidates
from yam_pld_rl.curator import reconstruct_intervals
from yam_pld_rl.intervention import InterventionSession
from yam_pld_rl.kinematics import YamKinematics


class FixtureFK:
    def position(self, joints):
        return np.array([0., 0., joints[0]])


def native_fixture(snapshot="fixture"):
    """Artificial original-recorder rows, not an executed robot trajectory."""
    c = contract()
    rec, _ = records(c, snapshot, attempts=2)
    proposals = [copy.deepcopy(r["proposal"]) for r in rec[3:6]]
    starts = [0, 8, 19, 28]  # native cadence is not d+8 fixed
    for b, p in enumerate(proposals):
        p.pop("attempt_id", None)
        p.update(start_tick=starts[b], end_tick=starts[b+1], delay_steps=0 if b == 0 else 2,
                 epoch=0, request_id=b+1, observation_id=100+b, decision_id=f"e0:r{b+1}",
                 run_id="synthetic", episode_id="ep", window_id="e0:w1", arm="left", mode="collect",
                 elapsed_seconds=starts[b]/30, gate={"active": True, "window_id": "e0:w1", "arm": "left"})
        if b:
            p["previous_command"] = proposals[b-1]["actions"][starts[b] - starts[b-1] - 1]
        p["scale"] = feasible_scale(p["base"], p["previous_command"], "left", p["delay_steps"], c).tolist()
        p["action"] = (np.asarray(p["scale"]) * p["sample_u"]).tolist()
        p["actions"] = decode(p["base"], p["action"], "left", p["delay_steps"], c).tolist()
        p["x"] = observation_vector(np.asarray(p["x"])[:1536], p["state"], p["base"], p["previous_command"],
                                    "left", p["delay_steps"], p["elapsed_seconds"], p["scale"], stats()).tolist()
        p.pop("end_tick", None)
    tail = {k: v for k, v in proposals[-1].items() if k not in {"x", "action", "sample_u", "scale", "actions", "arm", "window_id"}}
    tail.update(start_tick=starts[-1], request_id=4, observation_id=103, decision_id="e0:r4",
                gate={"active": False, "ended": {"window_id": "e0:w1", "reason": "lift_guard"}})
    proposals.append(tail)
    tokens = [{"epoch": 0, "request_id": p["request_id"], "observation_id": p["observation_id"],
               "observation_policy_tick": p["start_tick"], "rtc_delay_steps": p["delay_steps"]} for p in proposals]
    rows = []
    for t in range(starts[-1]+1):
        block = min(np.searchsorted(starts, t, side="right")-1, 2)
        p = proposals[block]
        source = block-1 if block and t - starts[block] < p["delay_steps"] else block
        origin, index = proposals[source], t - starts[source]
        state = np.zeros(14)
        state[6], state[13] = (.9 if t < 6 else .4), .9
        state[0] = .1 if t >= 22 else 0
        state[1] = 0 if t >= 22 else .5
        d = {"obs_id": 1000+t, "policy_selection": {"fusion": "rtc", "request": tokens[source],
             "target_tick": t, "model_index": index}}
        if t in starts:
            b = starts.index(t)
            q = proposals[b]
            d["obs_id"] = q["observation_id"]
            d["policy_reply"] = {"token": tokens[b], "pld": q, "actions": q.get("actions", q["base"]),
                                 "discarded": False, "error": None}
        rows.append({"time": 10000+t/30, "tick": t, "epoch": 0, "episode_id": "ep", "details": d,
                     "measured_state": state.tolist(), "measured_state_valid": True, "observation_valid": True,
                     "submitted_action": origin["actions"][index], "submitted_valid": True,
                     "is_intervention": False, "intervention_valid": True, "raw_ref": f"fixture/row{t}",
                     "camera_times": [10000+t/30]*3, "camera_valid": [True]*3, "video_indices": [t]*3,
                     "videos": {v: f"fixture/{v}.mp4" for v in ("top", "left", "right")}})
    candidates = physical_candidates(rows, FixtureFK())
    for candidate in candidates:
        candidate["candidate_sha256"] = fingerprint(candidate)
    candidate = candidates[0]
    evidence = [f["row"] for f in candidate["frames"] if f["lift_from_closure_m"] >= .05]
    reviews = [{"candidate_id": candidate["candidate_id"], "candidate_sha256": candidate["candidate_sha256"],
                "status": "held", "approved": True, "reviewer": "synthetic oracle", "evidence_rows": evidence}]
    return c, rows, candidates, reviews


class CuratorChecks(unittest.TestCase):
    def test_native_cadence_prefix_and_reward(self):
        c, rows, candidates, reviews = native_fixture()
        windows, audit = reconstruct_intervals(rows)
        self.assertFalse(audit)
        linked, labels, audit = associate_rewards(windows, candidates, reviews, rows, "train")
        self.assertFalse(audit)
        self.assertEqual([r["receipt"]["end_tick"]-r["receipt"]["start_tick"] for r in linked], [8, 11, 9])
        data, audit = compile_replay(linked, labels, c, {"fixture": {"contract": c, "encoder_sha256": "fixture", "normalization": stats()}})
        self.assertEqual(len(data["x"]), 3)
        self.assertEqual(data["reward"].sum(), 1)
        self.assertTrue(data["realized_nonzero"].all())
        self.assertFalse(audit["excluded"])
        self.assertNotIn("attempt_id", rows[0]["details"]["policy_reply"]["pld"])

    def test_bad_origin_hil_and_canceled_do_not_reconnect(self):
        for mutation in ("source", "hil", "cancel", "submission"):
            _, rows, candidates, reviews = native_fixture()
            if mutation == "source":
                rows[8]["details"]["policy_selection"]["request"] = rows[8]["details"]["policy_reply"]["token"]
            elif mutation == "hil":
                rows[10]["is_intervention"] = True
            elif mutation == "cancel":
                rows[8]["details"]["policy_reply"]["discarded"] = True
            else:
                rows[10]["submitted_action"] = np.ones(14).tolist()
            windows, rejected = reconstruct_intervals(rows)
            self.assertTrue(rejected)
            linked, labels, audit = associate_rewards(windows, candidates, reviews, rows, "train")
            self.assertFalse(linked)
            self.assertFalse(labels)
            self.assertTrue(audit)

    def test_review_release_missing_handoff_and_late_outcome(self):
        for mutation in ("hash", "handoff", "late", "gap"):
            _, rows, candidates, reviews = native_fixture()
            if mutation == "hash":
                reviews[0]["candidate_sha256"] = "stale"
            elif mutation == "handoff":
                rows[-1]["details"]["policy_reply"]["pld"]["gate"]["ended"] = None
            elif mutation == "gap":
                rows[-1]["details"]["policy_reply"]["pld"]["gate"]["ended"]["reason"] = "observation_gap"
            else:
                reviews[0]["evidence_rows"] = [29]
            windows, _ = reconstruct_intervals(rows)
            linked, _, audit = associate_rewards(windows, candidates, reviews, rows, "train")
            self.assertFalse(linked)
            self.assertTrue(audit)

    def test_gate_is_causal_and_connection_scoped(self):
        c = contract()
        first, second = InterventionSession(c), InterventionSession(c)
        state = np.zeros(14)
        state[[6,13]] = .9
        base = np.tile(state, (50,1))
        base[10:,6] = .4
        base[5:,13] = .4
        request = {"run_id": "r", "episode_id": "e", "epoch": 0, "mode": "collect"}
        def select(session, tick, z=0):
            return session.select(request, {"observation_policy_tick": tick}, state, base, {"left": z,"right": z})
        gate = select(first, 0)
        self.assertEqual(gate["arm"], "right")
        self.assertIsNone(second.active)
        state[13] = .4
        select(first, 5)
        self.assertTrue(select(first, 10, .06)["active"])
        self.assertEqual(select(first, 15, .06)["ended"]["reason"], "lift_guard")
        state[13] = .9
        request["epoch"] = 1
        self.assertTrue(select(first, 0)["window_id"].startswith("e1:s"))
        with self.assertRaises(ValueError):
            select(first, 0)
        first.reset()
        base[5:,13] = .9
        self.assertEqual(select(first, 0)["arm"], "left")
        base[5:,13] = .4
        base[5:,6] = .4
        second.reset()
        self.assertEqual(select(second, 0)["arm"], "left")

    def test_fk_pins_geometry(self):
        fk = YamKinematics(ROOT / "third_party/yam-fk-model/yam_linear4310.xml")
        point = fk.position(np.zeros(6))
        self.assertTrue(np.isfinite(point).all())
        self.assertTrue(-1 < point[2] < 1)


if __name__ == "__main__":
    unittest.main()
