"""CPU geometry equivalence checks; no model, learner or robot I/O."""
# ruff: noqa: E402, PT009, PT027
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from yam_pld_rl.kinematics import YamKinematics


class KinematicsChecks(unittest.TestCase):
    def setUp(self):
        self.fk = YamKinematics(ROOT / "third_party/yam-fk-model/yam_linear4310.xml")

    def test_batch_matches_scalar_across_joint_range(self):
        rng = np.random.default_rng(721)
        poses = rng.uniform(-np.pi, np.pi, (1024, 6))
        reference = np.asarray([self.fk.position(q) for q in poses])
        np.testing.assert_allclose(self.fk.positions(poses), reference, rtol=0, atol=2e-15)

    def test_gate_order_and_legacy_no_scope(self):
        rng = np.random.default_rng(54)
        state, base = rng.normal(size=14), rng.normal(size=(50, 14))
        scoped = self.fk.gate_heights(state, base, {"grasp_z_range_m": {"left": [-.1, .1]}})
        for arm, offset in (("left", 0), ("right", 7)):
            self.assertAlmostEqual(scoped[arm], self.fk.position(state[offset:offset+6])[2], places=14)
            expected = [self.fk.position(row[offset:offset+6])[2] for row in base]
            np.testing.assert_allclose(scoped["planned_z_m"][arm], expected, rtol=0, atol=2e-15)
        self.assertEqual(self.fk.gate_heights(state, base, {}), self.fk.heights(state))

    def test_batch_empty_and_invalid_inputs(self):
        self.assertEqual(self.fk.positions(np.empty((0, 6))).shape, (0, 3))
        for bad in (np.zeros(6), np.zeros((3, 7)), np.full((1, 6), np.nan)):
            with self.assertRaises(ValueError):
                self.fk.positions(bad)

    def test_batch_leaves_input_unchanged(self):
        poses = np.arange(120, dtype=np.float64).reshape(20, 6) / 100
        original = poses.copy()
        self.fk.positions(poses)
        np.testing.assert_array_equal(poses, original)


if __name__ == "__main__":
    unittest.main()
