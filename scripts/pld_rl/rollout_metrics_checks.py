"""Pure data-metric checks; no models, optimizer or robot connection."""

# ruff: noqa: PT009, PT027
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
from yam_pld_rl.rollout_metrics import episode_metrics


class RolloutMetricChecks(unittest.TestCase):
    def test_completed_round_rewards_fewer_attempts(self):
        a = episode_metrics(initial_objects=10, completed_objects=10, attempts=12, duration_s=60)
        b = episode_metrics(initial_objects=10, completed_objects=10, attempts=20, duration_s=60)
        self.assertGreater(a["completion_weighted_attempt_efficiency"], b["completion_weighted_attempt_efficiency"])
        self.assertEqual(a["attempts_if_complete"], 12)

    def test_early_stop_is_not_more_efficient(self):
        a = episode_metrics(initial_objects=10, completed_objects=1, attempts=1, duration_s=5)
        self.assertEqual(a["completion_weighted_attempt_efficiency"], 0)
        self.assertIsNone(a["attempts_if_complete"])
        self.assertEqual(a["completion_fraction"], .1)
        self.assertEqual(a["correct_completions_per_attempt"], 1)

    def test_unknown_counts_are_not_guessed(self):
        a = episode_metrics(initial_objects=10, completed_objects=None, attempts=None, duration_s=5)
        self.assertIsNone(a["completion_weighted_attempt_efficiency"])
        with self.assertRaises(ValueError):
            episode_metrics(initial_objects=10, completed_objects=10, attempts=0, duration_s=5)


if __name__ == "__main__":
    unittest.main()
