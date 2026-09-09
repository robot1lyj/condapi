"""Framework-independent, append-only scalar metric interchange (single writer)."""

import json
import math
from pathlib import Path


def write_metrics(path, step, metrics):
    """Write one complete event; callers own rank-zero and run-directory selection."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if not isinstance(metrics, dict) or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for key, value in metrics.items()
    ):
        raise ValueError("metrics must be named finite scalars")
    payload = json.dumps({"schema_version": 1, "step": step, "metrics": metrics}, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
