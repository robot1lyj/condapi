"""Pure-array contracts, reward bookkeeping, and replay sampling."""

from __future__ import annotations

import numpy as np


def progress_reward(initial_remaining, newly_correct, wrong, unplaced):
    counts = (initial_remaining, newly_correct, wrong, unplaced)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in counts):
        raise ValueError("counts must be integers")
    if initial_remaining <= 0 or min(counts[1:]) < 0 or sum(counts[1:]) != initial_remaining:
        raise ValueError("correct + wrong + unplaced must equal initially remaining bricks")
    return newly_correct / initial_remaining


def discount_seconds(dt, gamma_per_second):
    dt = np.asarray(dt, dtype=np.float64)
    if not 0 < gamma_per_second <= 1 or not np.isfinite(dt).all() or np.any(dt < 0):
        raise ValueError("invalid elapsed time or per-second discount")
    return np.power(gamma_per_second, dt).astype(np.float32)


def bounded_target(base, residual, bounds, lower, upper, *, committed=False):
    """Physical joints in rad; grippers normalized. Never alter RTC commitments."""
    base, residual, bounds, lower, upper = map(np.asarray, (base, residual, bounds, lower, upper))
    if base.shape[-1] != 14 or residual.shape != base.shape or any(x.shape != (14,) for x in (bounds, lower, upper)):
        raise ValueError("expected YAM 14D actions and per-coordinate physical bounds")
    if any(not np.isfinite(x).all() for x in (base, residual, bounds, lower, upper)):
        raise ValueError("nonfinite action contract")
    if np.any(bounds < 0) or np.any(lower >= upper) or np.any(np.abs(residual) > 1 + 1e-6):
        raise ValueError("invalid residual/action bounds")
    if np.any(base < lower) or np.any(base > upper):
        raise ValueError("base violates reviewed physical limits")
    proposed = np.clip(base + bounds * residual, lower, upper)
    mask = np.asarray(committed, dtype=bool)
    if mask.ndim:
        mask = mask[..., None]
    return np.where(mask, base, proposed)


def sample_episode_balanced(episode_lengths, size, rng):
    """A slow HIL episode must not dominate merely because it has more frames."""
    lengths = np.asarray(episode_lengths)
    if not len(lengths) or np.any(lengths <= 0) or size <= 0:
        raise ValueError("empty replay pool")
    episodes = rng.integers(0, len(lengths), size=size)
    rows = np.array([rng.integers(lengths[i]) for i in episodes])
    return episodes, rows
