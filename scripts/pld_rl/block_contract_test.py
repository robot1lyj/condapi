"""Array/receipt checks only; safe on workstation, never starts a learner."""

# ruff: noqa: PT009, PT027 -- stdlib unittest keeps CPU contract checks dependency-light.

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from block_fixtures import contract
from block_fixtures import records
from block_fixtures import stats
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_scale
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.block_replay import compile_replay


class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.c = contract()
        self.snapshots = {"fixture": {"contract": self.c, "encoder_sha256": "fixture", "normalization": stats()}}
        self.records, self.labels = records(self.c, "fixture")

    def test_zero_prefix_other_arm_and_bounds(self):
        rng = np.random.default_rng(55)
        base = np.full((50, 14), 0.12345678901234, dtype=np.float64)
        for arm in ("left", "right"):
            for delay in (0, 1, 10):
                scale = feasible_scale(base, base[0], arm, delay, self.c)
                zero = decode(base, np.zeros(7), arm, delay, self.c)
                self.assertEqual(zero.tobytes(), base.tobytes())
                for _ in range(100):
                    out = decode(base, scale * rng.uniform(-1, 1, 7), arm, delay, self.c)
                    self.assertEqual(out[:delay].tobytes(), base[:delay].tobytes())
                    other = slice(7, 14) if arm == "left" else slice(0, 7)
                    self.assertEqual(out[:, other].tobytes(), base[:, other].tobytes())
                    self.assertLessEqual(np.abs(np.diff(out, axis=0)).max(), 0.2)
                    self.assertLessEqual(np.abs(out - base).max(), 0.010000001)

    def test_near_limit_scale_reduces_without_clipping_zero(self):
        base = np.full((50, 14), 1.999, dtype=np.float64)
        scale = feasible_scale(base, base[0], "left", 3, self.c)
        self.assertLess(scale.max(), 0.11)
        out = decode(base, scale, "left", 3, self.c)
        self.assertLessEqual(out.max(), 2.000001)

    def test_missing_hardware_contract_is_rejected(self):
        self.c["lower"] = None
        with self.assertRaises(ValueError):
            validate_contract(self.c)

    def test_reward_only_once_and_dt(self):
        data, audit = compile_replay(self.records, self.labels, self.c, self.snapshots)
        self.assertEqual(audit["transitions"], 24)
        self.assertEqual(data["reward"].sum(), 4)
        self.assertTrue(np.all(data["reward"][data["terminal"] == 0] == 0))
        self.assertTrue(np.allclose(data["discount"], .99 ** data["dt"]))
        self.assertTrue(np.all(data["nx"][data["terminal"] == 1] == 0))

    def test_bad_execution_is_not_silently_relabeled(self):
        for field, value in (("client_modified", True), ("human_intervention", True), ("status", "canceled")):
            changed = copy.deepcopy(self.records)
            changed[0]["receipt"][field] = value
            data, audit = compile_replay(changed, self.labels, self.c, self.snapshots)
            self.assertEqual(len(data["reward"]), 21)
            self.assertEqual(len(audit["excluded"]), 1)
        changed = copy.deepcopy(self.records)
        changed[0]["receipt"]["submitted_actions"][2][1] += .01
        _, audit = compile_replay(changed, self.labels, self.c, self.snapshots)
        self.assertIn("actual actions differ", audit["excluded"][0]["reason"])

    def test_no_lift_and_wrong_endpoint_are_not_failures(self):
        for field, value in (("lift_m", .01), ("outcome_tick", 0), ("status", "unscored")):
            labels = copy.deepcopy(self.labels)
            labels[0][field] = value
            _, audit = compile_replay(self.records, labels, self.c, self.snapshots)
            self.assertEqual(len(audit["excluded"]), 1)

    def test_changed_observation_and_duplicates_rejected(self):
        changed = copy.deepcopy(self.records)
        changed[0]["proposal"]["x"][1600] += 1
        _, audit = compile_replay(changed, self.labels, self.c, self.snapshots)
        self.assertIn("causal observation", audit["excluded"][0]["reason"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compile_replay(self.records + self.records[:1], self.labels, self.c, self.snapshots)


if __name__ == "__main__":
    unittest.main()
