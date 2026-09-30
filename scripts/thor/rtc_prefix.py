"""Preserve already committed SDK targets without a float32 round trip."""

import numpy as np


def physical_prefix_array(value, delay):
    prefix = np.asarray(value)
    if delay == 0 and prefix.size == 0:
        prefix = prefix.reshape(0, 14)
    if prefix.shape != (delay, 14) or prefix.dtype.kind not in "fiu" or not np.isfinite(prefix).all():
        raise ValueError("RTC committed_actions must be finite [delay_steps,14] absolute targets")
    return prefix


def restore_physical_prefix(actions, prefix):
    if not len(prefix):
        return actions
    result = actions.astype(np.result_type(actions.dtype, prefix.dtype), copy=True)
    result[: len(prefix)] = prefix
    if not np.array_equal(result[: len(prefix)], prefix):
        raise ValueError("RTC committed targets cannot be preserved exactly")
    return result
