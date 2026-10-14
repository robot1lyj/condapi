"""CPU command-planner checks. No training and no connection to a robot."""

# ruff: noqa: PT009, PT027, PT019
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from block_fixtures import stats
from collector_checks import parent
from grasp_v3_checks import modern_recording
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import window_context
from yam_pld_rl.block_replay import compile_replay
from yam_pld_rl.collector import smooth_manifest
from yam_pld_rl.curator import associate_rewards
from yam_pld_rl.intervention import InterventionSession
from yam_pld_rl.trajectory import CorrelatedNoise
from yam_pld_rl.trajectory import planning_context
from yam_pld_rl.trajectory import trajectory_decode


class SimpleFK:
    def position(self, q):
        return np.asarray([0.2, 0.0, q[0]])

    def positions(self, qs):
        return np.asarray([[0.2, 0.0, q[0]] for q in qs]).reshape(-1, 3)


def manifest():
    p = parent()
    p["contract"].update(
        grasp_z_range_m={a: [-0.1, 0.06] for a in ("left", "right")}, grasp_scope_evidence="synthetic fixture"
    )
    return smooth_manifest(p, parent_snapshot="1" * 64)


class TrajectoryChecks(unittest.TestCase):
    def test_noise_stationary_and_can_reverse(self):
        noise = CorrelatedNoise()
        first = noise.advance("a", 0, 1 / 30, 0.5, np.full(7, -1.0))
        second = noise.advance("a", 10, 1 / 30, 0.5, np.full(7, 2.0))
        self.assertTrue(np.all(first["value"] < 0))
        self.assertTrue(np.all(second["value"] > 0))
        np.testing.assert_array_equal(noise.advance("b", 20, 1 / 30, 0.5, np.ones(7))["value"], np.ones(7))
        rng = np.random.default_rng(391)
        values = [noise.advance("c", i * 10, 1 / 30, 0.5, rng.normal(size=7))["value"] for i in range(6000)]
        self.assertTrue(np.all(np.abs(np.var(values, axis=0) - 1) < 0.1))
        with self.assertRaises(ValueError):
            noise.advance("c", 59990, 1 / 30, 0.5, np.zeros(7))

    @patch("yam_pld_rl.trajectory.model_fk", return_value=SimpleFK())
    def test_prefix_radius_slew_and_reverse(self, _fk):
        c = manifest()["contract"]
        b = np.zeros((50, 14), float)
        b[:, [6, 13]] = 0.9
        gate = {"apply_residual": True, "close_tick": None}
        ctx = planning_context(b, b[0], b[0], gate, "left", 10, c)
        self.assertIsNone(ctx["reason"])
        pos, neg = feasible_bounds(b, b[0], "left", 10, c, context=ctx)
        self.assertGreater(pos[:6].min(), 0.95)
        for direction in (-1.0, 1.0):
            a = map_action(np.full(7, direction * 0.8), pos, neg)
            out = decode(b, a, "left", 10, c, context=ctx)
            np.testing.assert_array_equal(out[:10], b[:10])
            np.testing.assert_array_equal(out[:, 7:], b[:, 7:])
            self.assertLessEqual(abs(out - b).max(), 0.08 + 1e-7)
            self.assertLessEqual(abs(np.diff(out - b, axis=0)).max(), 0.02 + 1e-7)
            self.assertEqual(np.sign(out[22, 0]), direction)
        # A new opposite target must replace, never accumulate, the old target.
        prev = decode(b, map_action(np.full(7, 0.7), pos, neg), "left", 10, c, context=ctx)
        next_base = b.copy()
        next_base[:10] = prev[10:20]
        ctx2 = planning_context(next_base, prev[9], b[0], gate, "left", 10, c)
        self.assertIsNone(ctx2["reason"])
        p, n = feasible_bounds(next_base, prev[9], "left", 10, c, context=ctx2)
        changed = decode(next_base, map_action(np.full(7, -0.8), p, n), "left", 10, c, context=ctx2)
        np.testing.assert_array_equal(changed[:10], next_base[:10])
        self.assertLess(changed[22, 0], 0)
        self.assertAlmostEqual(changed[10, 0] - changed[9, 0], changed[9, 0] - changed[8, 0], places=7)

    @patch("yam_pld_rl.trajectory.model_fk", return_value=SimpleFK())
    def test_future_lift_release_and_low_floor(self, _fk):
        c = manifest()["contract"]
        b = np.zeros((50, 14))
        b[:, [6, 13]] = 0.5
        gate = {"apply_residual": True, "close_tick": 0, "close_z": 0.0}
        for cause in ("lift", "release"):
            x = b.copy()
            if cause == "lift":
                x[:, 0] = np.clip((np.arange(50) - 22) * 0.01, 0, 0.08)
            else:
                x[:, 6] = 0.5 + np.clip((np.arange(50) - 25) * 0.05, 0, 0.4)
            ctx = planning_context(x, x[0], x[0], gate, "left", 10, c)
            self.assertEqual(ctx["end_index"], 27 if cause == "lift" else 28)
            p, n = feasible_bounds(x, x[0], "left", 10, c, context=ctx)
            result = decode(x, map_action(np.full(7, 0.5), p, n), "left", 10, c, context=ctx)
            np.testing.assert_array_equal(result[28:], x[28:])
        ctx = planning_context(b, b[0], b[0], gate, "left", 10, c)
        c["grasp_z_range_m"]["left"][0] = -0.01
        out, projection = trajectory_decode(b, np.full(7, -0.9), "left", c, ctx)
        self.assertTrue(np.all(out[:, 0] >= -0.01 - 1e-8))
        self.assertLess(projection["factor"], 1.0)
        gate["apply_residual"] = False
        ctx = planning_context(b, b[0], b[0], gate, "left", 10, c)
        np.testing.assert_array_equal(decode(b, np.zeros(7), "left", 10, c, context=ctx), b)

    def test_entry_over_container_is_blocked(self):
        c = manifest()["contract"]
        session = InterventionSession(c)
        q = np.zeros(14)
        q[[6, 13]] = 0.9
        b = np.tile(q, (50, 1))
        b[20:, 6] = 0.2
        h = {
            "left": 0.0,
            "right": 0.0,
            "planned_z_m": {"left": [0.0] * 50, "right": [0.0] * 50},
            "measured_xyz": {"left": [0.5, 0.0, 0.0], "right": [0.5, 0.0, 0.0]},
            "planned_xyz": {"left": [[0.5, 0.0, 0.0]] * 50, "right": [[0.5, 0.0, 0.0]] * 50},
        }
        g = session.select(
            {"run_id": "r", "episode_id": "e", "epoch": 0, "mode": "collect"},
            {"observation_policy_tick": 0, "delay_steps": 10},
            q,
            b,
            h,
        )
        self.assertFalse(g["active"])

    @patch("yam_pld_rl.trajectory.model_fk", return_value=SimpleFK())
    def test_native_replay_reconstructs_new_decoder(self, _fk):
        m = manifest()
        for key in ("contract", "actor_contract", "eval_contract"):
            m[key]["baseline_transform"] = "sdk_gripper_endpoints_v1"
        c, rows, records, candidates, reviews = modern_recording(contract_override=m["contract"])
        linked, labels, errors = associate_rewards(records, candidates, reviews, rows, "train")
        self.assertFalse(errors)
        ac = m["actor_contract"]
        for row in linked:
            p = row["proposal"]
            ap, an = feasible_bounds(p["base"], p["previous_command"], p["arm"], p["delay_steps"], ac)
            if not p["gate"]["apply_residual"]:
                ap, an = np.zeros(7), np.zeros(7)
            ax = observation_vector(
                np.asarray(p["x"])[:1536],
                p["state"],
                p["base"],
                p["previous_command"],
                p["arm"],
                p["delay_steps"],
                p["elapsed_seconds"],
                ap,
                stats(),
                negative_scale=an,
                context=window_context(p["gate"], p["start_tick"], p["base_z_m"][p["arm"]], 1 / 30),
            )
            eta = np.arctanh(np.asarray(p["sample_u"])) / np.asarray(m["collector"]["latent_std"])
            p.update(
                raw_base=copy.deepcopy(p["base"]),
                physical_base=copy.deepcopy(p["base"]),
                physical_actions=copy.deepcopy(p["actions"]),
                wire_actions=copy.deepcopy(p["actions"]),
                execution_contract_sha256=fingerprint(c),
                actor_context=ax[2264:],
                actor_mean_u=np.zeros(7),
                actor_scale=ap,
                actor_negative_scale=an,
                actor_physical_center=np.zeros(7),
                collector_latent_center=np.zeros(7),
                collector_latent_std=m["collector"]["latent_std"],
                collector_noise={"rho": 0.0, "previous": np.zeros(7), "innovation": eta, "value": eta},
            )
        data, audit = compile_replay(linked, labels, c, {"fixture": m})
        self.assertFalse(audit["excluded"])
        self.assertEqual(data["reward"].sum(), 1)
        # Floating reconstruction can differ by one ULP, but the recorded wire
        # and physical targets are identical. Preserve exact prefix checking.
        rounded = copy.deepcopy(linked)
        pp = next(r["proposal"] for r in rounded if r["proposal"]["delay_steps"] > 0)
        d = pp["delay_steps"]
        for key in ("actions", "physical_actions", "wire_actions"):
            pp[key][d][1] = float(np.nextafter(np.float64(pp[key][d][1]), np.inf))
        self.assertFalse(compile_replay(rounded, labels, c, {"fixture": m})[1]["excluded"])
        pp["wire_actions"][0][0] += 1e-6
        with self.assertRaisesRegex(ValueError, "wire plan changed"):
            compile_replay(rounded, labels, c, {"fixture": m})
        corrupted = copy.deepcopy(linked)
        corrupted[0]["proposal"]["trajectory_context"]["end_index"] += 1
        with self.assertRaisesRegex(ValueError, "trajectory context"):
            compile_replay(corrupted, labels, c, {"fixture": m})


if __name__ == "__main__":
    unittest.main()
