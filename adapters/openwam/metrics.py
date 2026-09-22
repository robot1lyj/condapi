"""Scalar-only bridge from OpenWAM's native log hook to the platform dashboard."""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/vla-platform/src"))
from vla_platform.metrics import write_metrics


def metrics_hook(original):
    def log_step(trainer, **values):
        original(trainer, **values)
        if trainer.accelerator is not None and not trainer.accelerator.is_main_process:
            return
        scalars = {key: float(value) for key, value in values["metrics"].items()}
        scalars.update(
            optimizer_step=int(values["opt_step"]),
            learning_rate=float(values["lr"]),
            epoch=int(values["epoch"]),
            microsteps_per_second=float(values["steps_per_sec"]),
        )
        # Preserve upstream failure diagnostics: never replace a nonfinite-loss run with a JSON serialization error.
        finite = {key: value for key, value in scalars.items() if math.isfinite(value)}
        finite["nonfinite_metric_count"] = len(scalars) - len(finite)
        write_metrics(Path(values["output_path"]) / "metrics.jsonl", int(values["global_step"]), finite)

    return log_step
