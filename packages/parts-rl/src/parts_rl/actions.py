"""Gripper-only chunk corrections; no hardware or controller implementation.

PARTS, arXiv:2609.21788v2, III-B and appendix D. A command proposal is
not execution evidence. The controller remains the sole action writer.
"""

from dataclasses import dataclass
import math

import numpy as np

GRIPPER_INDEX = {"left": 6, "right": 13}


@dataclass(frozen=True)
class ActionContract:
    arm: str
    base_id: str
    feature_id: str
    calibration_id: str
    horizon: int = 50
    # The paper's LEGO openness bound, not a joint angle or force bound.
    bound: float = 0.50
    # Explicit engineering choices; the paper does not publish their values.
    smoothing_weight: float = 1.0

    def __post_init__(self):
        if self.arm not in GRIPPER_INDEX or self.horizon != 50:
            raise ValueError("expected a YAM left/right H50 gripper contract")
        if not all(isinstance(x, str) and x for x in (self.base_id, self.feature_id, self.calibration_id)):
            raise ValueError("base, feature and calibrated openness identities are required")
        if not math.isfinite(self.bound) or not 0 < self.bound <= 0.50:
            raise ValueError("reviewed openness bound must be in (0, 0.50]")
        if not math.isfinite(self.smoothing_weight) or not 0 < self.smoothing_weight <= 1:
            raise ValueError("invalid causal smoothing weight")


def project_chunk(base, residual, editable, contract, *, committed_prefix, previous_u=0.0):
    """Return proposed targets and *post-projection* normalized corrections.

    The full chunk's original dtype and all twelve joint coordinates stay
    unchanged. committed_prefix is the actual physical RTC commitment, which
    the base sampler must already have reproduced exactly. A future inactive
    row is base-only; a later controller gate must also zero stale proposals.
    previous_u is the smoothing state immediately before the editable suffix
    (after any commitment), not a state to replay through committed rows.
    """
    base = np.asarray(base)
    residual = np.asarray(residual, dtype=np.float64)
    editable = np.asarray(editable)
    prefix = np.asarray(committed_prefix)
    horizon = contract.horizon
    if base.shape != (horizon, 14) or base.dtype.kind != "f":
        raise ValueError("expected floating physical H50x14 base targets")
    if residual.shape != (horizon, 1) or editable.shape != (horizon,) or editable.dtype != np.bool_:
        raise ValueError("expected H50x1 residual and boolean editable mask")
    if prefix.ndim != 2 or prefix.shape[1:] != (14,) or not 0 <= len(prefix) <= horizon:
        raise ValueError("invalid physical RTC prefix")
    if not all(np.isfinite(x).all() for x in (base, residual, prefix)) or not math.isfinite(previous_u):
        raise ValueError("nonfinite action evidence")
    if np.any(np.abs(residual) > 1 + 1e-6) or abs(previous_u) > 1:
        raise ValueError("normalized residual outside [-1, 1]")
    if np.any(editable[: len(prefix)]) or not np.array_equal(base[: len(prefix)], prefix):
        raise ValueError("RTC committed prefix must remain exact and noneditable")
    index = GRIPPER_INDEX[contract.arm]
    if np.any((base[:, index] < 0) | (base[:, index] > 1)):
        raise ValueError("base openness requires the reviewed 0-closed/1-open projection first")
    target = base.copy()
    applied = np.zeros((horizon, 1), dtype=np.float64)
    last = float(previous_u)
    for row in range(horizon):
        if row < len(prefix):
            continue
        if not editable[row]:
            last = 0.0
            continue
        last = contract.smoothing_weight * float(residual[row, 0]) + (1 - contract.smoothing_weight) * last
        target[row, index] = np.clip(base[row, index] + contract.bound * last, 0, 1)
        applied[row, 0] = (float(target[row, index]) - float(base[row, index])) / contract.bound
    return target, applied


def explore(mean, mask, *, std, rng):
    """PARTS normalized-space Gaussian exploration, clipped after adding noise."""
    mean, mask = np.asarray(mean), np.asarray(mask)
    if mean.ndim != 2 or mean.shape[1] != 1 or mask.shape != mean.shape[:1] or mask.dtype != np.bool_:
        raise ValueError("invalid gripper chunk/mask")
    if not math.isfinite(std) or std < 0 or not np.isfinite(mean).all() or np.any(np.abs(mean) > 1):
        raise ValueError("invalid mean or exploration scale")
    return np.where(mask[:, None], np.clip(mean + rng.normal(0, std, mean.shape), -1, 1), 0)
