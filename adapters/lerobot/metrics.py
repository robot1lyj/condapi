"""Capture native full-precision tracker events without replacing the trainer."""

import logging
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/vla-platform/src"))
from vla_platform.metrics import write_metrics


class TrackerMetrics(logging.Filter):
    """Root filters survive LeRobot init_logging's handler reset; no text parsing."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.failed = False

    def filter(self, record):
        tracker = record.msg
        if (type(tracker).__module__, type(tracker).__name__) != ("lerobot.utils.logging_utils", "MetricsTracker"):
            return True
        try:
            values = tracker.to_dict()
            step = values.pop("steps")
            for native, canonical in {
                "step_s": "step_seconds",
                "samples_per_s": "samples_per_second",
                "gpu_mem_gb": "gpu_peak_memory_gib",
                "lr": "learning_rate",
            }.items():
                if native in values:
                    values[canonical] = values.pop(native)
            write_metrics(self.path, step, values)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Telemetry must not interrupt an otherwise healthy training process.
            if not self.failed:
                warnings.warn(f"Dashboard metrics unavailable: {exc}", RuntimeWarning, stacklevel=2)
            self.failed = True
        return True
