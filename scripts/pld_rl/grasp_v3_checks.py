"""CPU array/recorder regression cases; never train or command hardware."""

# ruff: noqa: E402, PT009, PT027
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from block_fixtures import contract
from block_fixtures import stats
from curator_checks import FixtureFK
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.block_contract import window_context
from yam_pld_rl.block_replay import compile_replay
from yam_pld_rl.curator import associate_rewards
from yam_pld_rl.curator import physical_candidates
from yam_pld_rl.curator import reconstruct_intervals
from yam_pld_rl.grasp_outcomes import attach_outcomes
from yam_pld_rl.intervention import DEFAULT_GATE
from yam_pld_rl.intervention import InterventionSession


def modern_contract():
    c = contract()
    c.update(gate=DEFAULT_GATE.copy(), action_mapping="signed_feasible_v1", reward_contract="drained_grasp_v1")
    c["lower"][6] = c["lower"][13] = 0.02
    c["upper"][6] = c["upper"][13] = 0.98
    c["max_target_step"][6] = c["max_target_step"][13] = 0.96
    validate_contract(c)
    return c


def modern_recording(kind="held", *, contract_override=None):
    c, rows, proposals, tokens = contract_override or modern_contract(), [], [], []
    session = InterventionSession(c)
    identity = {"run_id": "synthetic-v3", "episode_id": "ep", "epoch": 0, "mode": "collect"}
    previous = np.zeros(14)
    previous[[6, 13]] = 0.9
    for tick in range(140):
        state = np.zeros(14)
        state[[6, 13]] = [0.9 if kind == "no_closure" or tick < 6 else 0.4, 0.9]
        state[0] = 0.1 if kind in {"held", "empty", "dropped"} and tick >= 20 else 0
        details = {"obs_id": 1000 + tick}
        if tick % 5 == 0:
            delay = 2 if proposals else 0
            base = np.tile(state, (50, 1))
            base[delay + 4 :, 6] = 0.4
            if proposals:
                base[:delay] = proposals[-1]["actions"][
                    tick - proposals[-1]["start_tick"] : tick - proposals[-1]["start_tick"] + delay
                ]
            gate = session.select(
                identity,
                {"observation_policy_tick": tick, "delay_steps": delay},
                state,
                base,
                {"left": state[0], "right": 0.0},
            )
            p = {
                **identity,
                "request_id": len(proposals) + 1,
                "observation_id": 1000 + tick,
                "decision_id": f"e0:r{len(proposals) + 1}",
                "schema": c["schema"],
                "snapshot_sha256": "fixture",
                "contract_sha256": fingerprint(c),
                "start_tick": tick,
                "delay_steps": delay,
                "base": base.tolist(),
                "state": state.tolist(),
                "base_z_m": {"left": float(state[0]), "right": 0.0},
                "previous_command": previous.tolist(),
                "actions": base.tolist(),
                "gate": copy.deepcopy(gate),
                "replay_eligible": True,
            }
            if gate["active"]:
                pos, neg = feasible_bounds(base, previous, gate["arm"], delay, c)
                if not gate["apply_residual"]:
                    pos, neg = np.zeros(7, np.float32), np.zeros(7, np.float32)
                u = np.full(7, 0.2, np.float32)
                a = map_action(u, pos, neg)
                p.update(
                    arm=gate["arm"],
                    window_id=gate["window_id"],
                    elapsed_seconds=gate["elapsed_seconds"],
                    scale=pos.tolist(),
                    negative_scale=neg.tolist(),
                    sample_u=u.tolist(),
                    action=a.tolist(),
                    x=observation_vector(
                        np.zeros(1536),
                        state,
                        base,
                        previous,
                        gate["arm"],
                        delay,
                        gate["elapsed_seconds"],
                        pos,
                        stats(),
                        negative_scale=neg,
                        context=window_context(gate, tick, state[0], 1 / 30),
                    ).tolist(),
                    actions=decode(base, a, gate["arm"], delay, c).tolist(),
                )
                session.note_proposal(tick, delay)
            token = {
                "epoch": 0,
                "request_id": p["request_id"],
                "observation_id": p["observation_id"],
                "observation_policy_tick": tick,
                "rtc_delay_steps": delay,
            }
            details["policy_reply"] = {
                "token": token,
                "pld": p,
                "actions": p["actions"],
                "discarded": False,
                "error": None,
            }
            proposals.append(p)
            tokens.append(token)
        source = max(i for i, p in enumerate(proposals) if p["start_tick"] + p["delay_steps"] <= tick)
        origin = proposals[source]
        k = tick - origin["start_tick"]
        submitted = np.asarray(origin["actions"][k])
        details["policy_selection"] = {
            "fusion": "rtc",
            "request": tokens[source],
            "target_tick": tick,
            "model_index": k,
        }
        rows.append(
            {
                "time": 10000 + tick / 30,
                "tick": tick,
                "epoch": 0,
                "episode_id": "ep",
                "details": details,
                "measured_state": state.tolist(),
                "observation_state": state.tolist(),
                "measured_state_valid": True,
                "observation_valid": True,
                "submitted_action": submitted.tolist(),
                "submitted_valid": True,
                "is_intervention": False,
                "intervention_valid": True,
                "raw_ref": f"fixture/row{tick}",
                "camera_times": [10000 + tick / 30] * 3,
                "camera_valid": [True] * 3,
                "video_indices": [tick] * 3,
                "videos": {v: f"fixture/{v}.mp4" for v in ("top", "left", "right")},
            }
        )
        previous = submitted
        if tick % 5 == 0 and proposals[-1]["gate"].get("ended"):
            break
    records, audit = reconstruct_intervals(rows)
    assert not audit, audit
    candidates = physical_candidates(rows, FixtureFK())
    attach_outcomes(candidates, records, rows, FixtureFK())
    for candidate in candidates:
        candidate["candidate_sha256"] = fingerprint(candidate)
    candidate = next(c for c in candidates if "outcome" in c)
    review = {
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "status": kind,
        "approved": True,
        "reviewer": "synthetic oracle",
        "grasp_intent_confirmed": True,
        "held_then_lost_confirmed": kind == "dropped",
        "evidence_rows": candidate["outcome"]["fresh_rows"][-2:],
    }
    return c, rows, records, candidates, [review]


class GraspV3Checks(unittest.TestCase):
    def test_multiple_windows_bind_each_own_endpoint(self):
        _, rows, records, candidates, reviews = modern_recording()
        shift = len(rows)
        extra_rows = copy.deepcopy(rows)
        for row in extra_rows:
            row["tick"] += shift
        extra_records = copy.deepcopy(records)
        for record in extra_records:
            proposal, receipt = record["proposal"], record["receipt"]
            proposal["start_tick"] += shift
            proposal["window_id"] += ":second"
            if "end_tick" in proposal:
                proposal["end_tick"] += shift
            receipt["end_tick"] += shift
        original = next(c for c in candidates if c.get("outcome"))
        extra = copy.deepcopy(original)
        extra["candidate_id"] += ":second"
        extra["close_tick"] += shift
        extra["outcome"]["window_id"] += ":second"
        extra["outcome"]["terminal_tick"] += shift
        for key in ("fresh_rows", "lifted_rows"):
            extra["outcome"][key] = [i + shift for i in extra["outcome"][key]]
        for frame in extra["frames"]:
            frame["row"] += shift
        review = {**reviews[0], "candidate_id": extra["candidate_id"],
                  "evidence_rows": [i + shift for i in reviews[0]["evidence_rows"]]}
        linked, labels, audit = associate_rewards(records + extra_records, [original, extra],
                                                  [*reviews, review], rows + extra_rows, "train")
        self.assertFalse(audit)
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(linked), len(records) * 2)
        self.assertEqual(labels[0]["terminal_tick"] + shift, labels[1]["terminal_tick"])

    def compile(self, case):
        c, rows, records, candidates, reviews = case
        linked, labels, audit = associate_rewards(records, candidates, reviews, rows, "train")
        self.assertFalse(audit)
        data, audit = compile_replay(
            linked, labels, c, {"fixture": {"contract": c, "normalization": stats(), "encoder_sha256": "fixture"}}
        )
        self.assertFalse(audit["excluded"])
        return data, labels

    def test_fast_lift_stop_drains_and_rewards_after_tail(self):
        case = modern_recording()
        data, labels = self.compile(case)
        proof = labels[0]["outcome_proof"]
        self.assertEqual(proof["reason"], "lift_guard")
        self.assertGreater(proof["terminal_tick"], proof["drain_until_tick"])
        self.assertGreater(proof["drain_until_tick"], proof["stop_tick"])
        self.assertEqual(data["reward"].sum(), 1)
        self.assertTrue(np.all(data["action"][-2:] == 0))
        self.assertTrue(np.all(data["negative_scale"][-2:] == 0))
        self.assertTrue(
            any(e["tick"] >= proof["stop_tick"] for r in case[2] for e in r["receipt"]["actor_effects_before_handoff"])
        )

    def test_no_lift_is_a_zero_goal_reward_not_missing_data(self):
        case = modern_recording("not_lifted")
        data, labels = self.compile(case)
        self.assertEqual(data["reward"].sum(), 0)
        self.assertLess(labels[0]["lift_m"], 0.05)
        self.assertEqual(labels[0]["outcome_proof"]["reason"], "closed_timeout")

    def test_no_closure_is_kept_with_reviewed_intent(self):
        data, labels = self.compile(modern_recording("no_closure"))
        self.assertEqual(data["reward"].sum(), 0)
        self.assertFalse(labels[0]["outcome_proof"]["close_confirmed"])

    def test_dropped_is_negative_even_after_reaching_height(self):
        data, labels = self.compile(modern_recording("dropped"))
        self.assertEqual(data["reward"].sum(), 0)
        self.assertGreater(labels[0]["outcome_proof"]["max_lift_m"], 0.05)

    def test_missing_data_or_unreviewed_intent_is_never_failure(self):
        for mutation in ("intent", "measured", "camera", "duplicate"):
            _, rows, records, candidates, reviews = modern_recording(
                "not_lifted" if mutation != "duplicate" else "held"
            )
            if mutation == "intent":
                reviews[0]["grasp_intent_confirmed"] = False
            elif mutation == "measured":
                rows[15]["measured_state_valid"] = False
                attach_outcomes(candidates, records, rows, FixtureFK())
            elif mutation == "camera":
                for row in rows:
                    row["camera_times"] = [9999.0] * 3
                attach_outcomes(candidates, records, rows, FixtureFK())
            else:
                for f in candidates[0]["frames"]:
                    f["views"]["left"]["video_frame_index"] = 1
            for candidate in candidates:
                candidate["candidate_sha256"] = fingerprint(candidate)
            reviews[0]["candidate_sha256"] = candidates[0]["candidate_sha256"]
            linked, labels, audit = associate_rewards(records, candidates, reviews, rows, "train")
            self.assertFalse(linked, mutation)
            self.assertFalse(labels, mutation)
            self.assertTrue(audit, mutation)

    def test_pre_tail_success_and_early_terminal_rejected(self):
        c, rows, records, candidates, reviews = modern_recording()
        reviews[0]["evidence_rows"] = [22]
        self.assertTrue(associate_rewards(records, candidates, reviews, rows, "train")[2])
        case = modern_recording()
        c, rows, records, candidates, reviews = case
        linked, labels, audit = associate_rewards(records, candidates, reviews, rows, "train")
        self.assertFalse(audit)
        labels[0]["outcome_proof"]["drain_until_tick"] = labels[0]["terminal_tick"]
        with self.assertRaisesRegex(ValueError, "no eligible replay"):
            compile_replay(
                linked, labels, c, {"fixture": {"contract": c, "normalization": stats(), "encoder_sha256": "fixture"}}
            )

    def test_signed_endpoints_retain_valid_direction_and_zero(self):
        c = modern_contract()
        for endpoint, sign in ((0.98, -1), (0.02, 1)):
            b = np.full((50, 14), 0.4)
            b[:, 6] = endpoint
            pos, neg = feasible_bounds(b, b[0], "left", 2, c)
            self.assertGreater((neg if sign < 0 else pos)[6], 0.9)
            u = np.zeros(7)
            u[6] = 0.9 * sign
            action = map_action(u, pos, neg)
            changed = decode(b, action, "left", 2, c)
            self.assertGreater(abs(changed[5, 6] - endpoint), 0.001)
            self.assertTrue(np.all(changed[:, 6] >= 0.02 - 1e-7))
            self.assertTrue(np.all(changed[:, 6] <= 0.98 + 1e-7))
            self.assertEqual(decode(b, map_action(np.zeros(7), pos, neg), "left", 2, c).tobytes(), b.tobytes())

    def test_signed_support_obeys_limits_and_slew(self):
        c = modern_contract()
        c["max_target_step"] = [0.05] * 14
        c["max_residual_step"] = [0.004] * 14
        rng = np.random.default_rng(818)
        for arm in ("left", "right"):
            for _ in range(50):
                b = np.full((50, 14), 0.4)
                b += np.cumsum(rng.uniform(-0.015, 0.015, (50, 14)), axis=0)
                pos, neg = feasible_bounds(b, b[0], arm, 2, c)
                a = map_action(rng.uniform(-1, 1, 7), pos, neg)
                out = decode(b, a, arm, 2, c)
                self.assertTrue(np.all(out >= np.asarray(c["lower"]) - 1e-7))
                self.assertTrue(np.all(out <= np.asarray(c["upper"]) + 1e-7))
                self.assertLessEqual(np.abs(np.diff(out, axis=0)).max(), 0.05 + 1e-7)

    def test_intervention_during_queue_tail_is_not_a_zero_reward(self):
        _, rows, _, _, _ = modern_recording("not_lifted")
        rows[-3]["is_intervention"] = True
        records, audit = reconstruct_intervals(rows)
        self.assertTrue(audit)
        candidates = physical_candidates(rows, FixtureFK())
        attach_outcomes(candidates, records, rows, FixtureFK())
        self.assertTrue(all("outcome" not in c for c in candidates))

    def test_new_arm_may_start_at_previous_settlement_without_overlap(self):
        c = modern_contract()
        session = InterventionSession(c)
        request = {"run_id": "r", "episode_id": "e", "epoch": 0, "mode": "collect"}
        state = np.zeros(14)
        state[[6, 13]] = 0.9
        for tick, g, z in ((0, 0.9, 0.0), (5, 0.4, 0.0), (10, 0.4, 0.1), (15, 0.4, 0.1)):
            state[6] = g
            base = np.tile(state, (50, 1))
            base[4:, 6] = 0.4
            if tick == 15:
                base[4:, 13] = 0.4
            result = session.select(
                request, {"observation_policy_tick": tick, "delay_steps": 0}, state, base, {"left": z, "right": 0.0}
            )
            if result["active"]:
                session.note_proposal(tick, 0)
        self.assertEqual(result["ended"]["arm"], "left")
        self.assertTrue(result["active"])
        self.assertEqual(result["arm"], "right")
        self.assertNotEqual(result["window_id"], result["ended"]["window_id"])

    def test_no_lift_outcome_is_exported_for_review(self):
        import av  # noqa: PLC0415
        from curate_recording import export_evidence  # noqa: PLC0415

        _, rows, _, candidates, _ = modern_recording("not_lifted")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "test.mp4"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                for _ in rows:
                    frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            for name in ("left", "right", "top"):
                (root / f"{name}.mp4").write_bytes(video.read_bytes())
            for candidate in candidates:
                for frame in candidate["frames"]:
                    for name, view in frame["views"].items():
                        view["source_video"] = str(root / f"{name}.mp4")
            output = root / "output"
            output.mkdir()
            export_evidence(candidates, output)
            exported = json.loads((output / "outcomes-prepared.json").read_text())
            self.assertEqual(len(exported), 1)
            self.assertEqual(exported[0]["proposed_status"], "not_lifted")
            self.assertFalse(exported[0]["approved"])
            self.assertTrue((output / exported[0]["montage"]).is_file())


if __name__ == "__main__":
    unittest.main()
