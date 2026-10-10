"""Condapi extension: configurable prefix sampling for native XR-1 training."""

import random


def validate_prefix_range(min_steps, max_steps, horizon=30):
    if (
        type(min_steps) is not int
        or type(max_steps) is not int
        or not 1 <= min_steps <= max_steps < horizon
    ):
        raise ValueError("async prefix must satisfy 1 <= min_steps <= max_steps < action horizon")


def sample_prefix_length(enabled, min_steps=1, max_steps=6, rng=random):
    validate_prefix_range(min_steps, max_steps)
    return rng.randint(min_steps, max_steps) if enabled and rng.random() < 0.5 else 0
