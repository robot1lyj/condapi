"""Array-only intervention regressions; no learner, camera or robot I/O."""
# ruff: noqa: E402, PT009
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from block_fixtures import contract
from yam_pld_rl.block_contract import physical_baseline
from yam_pld_rl.block_contract import wire_plan
from yam_pld_rl.grasp_events import relative_physical_events
from yam_pld_rl.intervention import DEFAULT_GATE
from yam_pld_rl.intervention import LEGACY_GATE
from yam_pld_rl.intervention import InterventionSession


class InterventionChecks(unittest.TestCase):
    def setUp(self):
        self.contract = contract()
        self.gate = InterventionSession(self.contract)
        self.tick = -10

    def step(self, g, z, *, future=None, delay=9, request=None, heights=None):
        self.tick += 10
        state = np.zeros(14)
        state[[6, 13]] = (g, .9)
        base = np.tile(state, (50, 1))
        if future is not None:
            base[delay + 4:, 6] = future
        identity = {"run_id": "r", "episode_id": "e", "epoch": 0, "mode": "probe"}
        return self.gate.select(request or identity, {"observation_policy_tick": self.tick, "delay_steps": delay},
                                state, base, heights or {"left": z, "right": .1})

    def test_partial_opening_can_enter_and_rearm_below_065(self):
        first = self.step(.64, .05, future=.3)
        self.assertTrue(first["active"])
        self.step(.4, .01)
        self.step(.3, .03)
        ended = self.step(.3, .08)
        self.assertEqual(ended["ended"]["reason"], "lift_guard")
        self.assertFalse(self.step(.31, .1, future=.1)["active"])
        self.assertTrue(self.step(.64, .06, future=.3)["active"])

    def test_wide_object_does_not_require_closed_062(self):
        self.step(.96, .08, future=.65)
        self.assertIsNotNone(self.step(.76, .03)["close_tick"])
        self.step(.68, .01)
        self.step(.68, .04)
        ended = self.step(.68, .08)
        self.assertEqual(ended["ended"]["reason"], "lift_guard")

    def test_initial_reopening_inside_forecast_is_not_false_exit(self):
        self.step(.38, .13, future=.24)
        self.step(.64, .07)
        result = self.step(.53, .03)
        self.assertTrue(result["active"])
        self.assertIsNotNone(result["close_tick"])
        self.assertIsNone(result["ended"])

    def test_early_close_tracks_bottom_of_subsequent_descent(self):
        self.step(.95, .15, future=.3)
        self.step(.75, .1)
        self.step(.5, .01)
        result = self.step(.4, .04)
        self.assertAlmostEqual(result["close_z"], .01)
        self.assertEqual(self.step(.4, .07)["ended"]["reason"], "lift_guard")

    def test_rising_trend_is_required_and_timeout_is_bounded(self):
        self.step(.9, .01, future=.3)
        self.step(.6, .01)
        self.assertTrue(self.step(.4, .1)["active"])
        self.assertTrue(self.step(.4, .1)["active"])
        for _ in range(45):
            result = self.step(.4, .1)
            if result["ended"]:
                break
        self.assertEqual(result["ended"]["reason"], "timeout")

    def test_prefix_only_forecast_cannot_trigger(self):
        state = np.zeros(14)
        state[[6, 13]] = .9
        base = np.tile(state, (50, 1))
        base[:9, 6] = .1
        result = self.gate.select({"run_id": "r", "episode_id": "e", "epoch": 0, "mode": "probe"},
                                  {"observation_policy_tick": 0, "delay_steps": 9}, state, base,
                                  {"left": .01, "right": .01})
        self.assertFalse(result["active"])

    def test_scope_rejects_high_closure_and_accepts_calibrated_plane(self):
        self.contract.update(grasp_z_range_m={"left": [-.01, .05], "right": [-.01, .05]})
        self.assertFalse(self.step(.9, .12, future=.3,
                                   heights={"left": .12, "right": .1,
                                            "planned_z_m": {"left": [.08]*50, "right": [.08]*50}})["active"])
        self.assertTrue(self.step(.9, .12, future=.3,
                                  heights={"left": .12, "right": .1,
                                           "planned_z_m": {"left": [.02]*50, "right": [.02]*50}})["active"])

    def test_legacy_contract_preserves_original_absolute_threshold(self):
        self.contract["gate"] = LEGACY_GATE.copy()
        self.assertFalse(self.step(.64, .05, future=.3)["active"])
        self.assertEqual(DEFAULT_GATE["version"], 3)

    def test_relative_segmentation_separates_retries_below_065(self):
        g = np.r_[np.full(5, .9), np.linspace(.9, .2, 10), np.full(10, .2),
                  np.linspace(.2, .64, 10), np.linspace(.64, .2, 10), np.full(10, .2)]
        n = len(g)
        events = relative_physical_events(np.arange(n)/30, g, np.zeros(n), np.ones(n, bool),
                                          np.zeros(n, int), np.arange(n))
        self.assertEqual(len(events), 2)
        self.assertLess(events[0]["end"], events[1]["close"])

    def test_sdk_baseline_preserves_prefix_and_joint_targets(self):
        self.contract.update(baseline_transform="sdk_gripper_endpoints_v1")
        self.contract["lower"][6] = self.contract["lower"][13] = .02
        self.contract["upper"][6] = self.contract["upper"][13] = .98
        raw = np.full((50, 14), .12345678901234)
        raw[9:, 6] = 1.01
        baseline = physical_baseline(raw, 9, self.contract)
        self.assertEqual(raw[:9].tobytes(), baseline[:9].tobytes())
        self.assertTrue(np.all(baseline[9:, 6] == .98))
        self.assertEqual(raw[:, :6].tobytes(), baseline[:, :6].tobytes())
        raw[0, 6] = 1.01
        baseline = physical_baseline(raw, 9, self.contract)
        wire = wire_plan(baseline, raw, 9)
        self.assertEqual(wire[:9].tobytes(), raw[:9].tobytes())
        self.assertEqual(baseline[0, 6], .98)


if __name__ == "__main__":
    unittest.main()
