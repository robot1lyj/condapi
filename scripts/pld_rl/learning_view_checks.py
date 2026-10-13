"""CPU array checks only: preserve physical endpoints and recorded returns."""
# ruff: noqa: PT009, PT027
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.learning_view import learning_view


def fixture():
    x = np.arange(12).reshape(4, 3).astype(np.float32)
    d = {"x": x, "nx": np.vstack((x[1:], np.zeros(3))),
         "action": np.ones((4, 7)) * .2, "scale": np.ones((4, 7)), "negative_scale": np.ones((4, 7)),
         "next_scale": np.ones((4, 7)), "next_negative_scale": np.ones((4, 7)),
         "reward": np.array([0., 0., 0., 1.]), "terminal": np.array([0., 0., 0., 1.]),
         "discount": np.array([.9, .8, .7, .6]), "dt": np.array([.1, .2, .3, .4]),
         "attempt": np.array(["a"]*4), "episode": np.array(["ep"]*4), "decision": np.arange(4),
         "train": np.ones(4, bool), "arm": np.zeros(4, bool)}
    for key in ("action", "scale", "negative_scale"):
        d[key][-2:] = 0
    d["next_scale"][-3:] = d["next_negative_scale"][-3:] = 0
    return d


class LearningViewChecks(unittest.TestCase):
    def test_collapse_preserves_return_and_endpoint(self):
        source = fixture()
        original = copy.deepcopy(source)
        out, audit = learning_view(source)
        self.assertEqual(len(out["x"]), 2)
        np.testing.assert_allclose(out["mc_return"], [.9*.8*.7, .8*.7])
        self.assertAlmostEqual(out["reward"][-1], .8*.7)
        self.assertAlmostEqual(out["discount"][-1], .8*.7*.6)
        self.assertAlmostEqual(out["dt"][-1], .9)
        np.testing.assert_array_equal(out["source_count"], [1, 3])
        np.testing.assert_array_equal(out["next_action"][0], source["action"][1])
        np.testing.assert_array_equal(out["attempt_outcome"], [1, 1])
        np.testing.assert_array_equal(out["nx"][-1], source["nx"][-1])
        self.assertTrue(audit["reward_endpoint_unchanged"])
        for key in source:
            np.testing.assert_array_equal(source[key], original[key])

    def test_no_collapse_and_failure(self):
        source = fixture()
        source["reward"][-1] = 0
        out, _ = learning_view(source, collapse_wait=False)
        self.assertEqual(len(out["x"]), 4)
        self.assertEqual(out["mc_return"].sum(), 0)
        self.assertEqual(out["attempt_outcome"].sum(), 0)

    def test_active_last_action_kept(self):
        source = fixture()
        source["scale"][-1] = 1
        source["action"][-1] = .1
        out, _ = learning_view(source)
        self.assertEqual(len(out["x"]), 4)

    def test_broken_chain_split_and_nonzero_wait_rejected(self):
        for key, index, value in (("nx", 0, 99), ("train", 1, False), ("action", 3, .2),
                                  ("terminal", 0, 1), ("reward", 0, 1)):
            source = fixture()
            source[key][index] = value
            with self.assertRaises(ValueError):
                learning_view(source)


if __name__ == "__main__":
    unittest.main()
