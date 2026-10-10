"""CPU-only checks for prefix sampling; never construct the XR-1 model."""

# Standard-library unittest keeps these checks independent of model environments.
# ruff: noqa: PT009, PT027

import importlib.util
from pathlib import Path
import random
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "third_party/xr1/mibot/utils/async_prefix.py"
SPEC = importlib.util.spec_from_file_location("xr1_async_prefix", SOURCE)
prefix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefix)


class PrefixSamplingTest(unittest.TestCase):
    def test_default_preserves_upstream_sampling(self):
        actual_rng, expected_rng = random.Random(42), random.Random(42)
        for _ in range(100):
            expected = expected_rng.randint(1, 6) if expected_rng.random() < 0.5 else 0
            self.assertEqual(prefix.sample_prefix_length(enabled=True, rng=actual_rng), expected)

    def test_expanded_range_includes_ten_and_zero(self):
        samples = {
            prefix.sample_prefix_length(enabled=True, min_steps=1, max_steps=10, rng=random.Random(seed))
            for seed in range(1000)
        }
        self.assertEqual(samples, set(range(11)))

    def test_disabled_preserves_rng_and_returns_zero(self):
        rng = random.Random(42)
        state = rng.getstate()
        self.assertEqual(prefix.sample_prefix_length(enabled=False, min_steps=1, max_steps=10, rng=rng), 0)
        self.assertEqual(state, rng.getstate())

    def test_invalid_range_rejected(self):
        for low, high in ((0, 10), (10, 1), (1, 30), (True, 10), (1, 10.0)):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                prefix.validate_prefix_range(low, high)


if __name__ == "__main__":
    unittest.main()
