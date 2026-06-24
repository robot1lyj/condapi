import atexit
import csv
import json
import math
import numbers
import pathlib
import re
from collections.abc import Mapping
from typing import Any


def _to_scalar(value: Any) -> int | float | str | bool | None:
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, numbers.Real | str | bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    return None


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "metric"


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values

    averaged = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        averaged.append(running_sum / min(index + 1, window))
    return averaged


class LocalMetricLogger:
    """Writes training metrics to jsonl/csv and renders offline curve images."""

    def __init__(self, run_dir: str | pathlib.Path, *, resume_step: int | None = None, plot_every_n_logs: int = 5, flush_on_log: bool = False):
        self.run_dir = pathlib.Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.plots_dir = self.metrics_dir / "plots"
        self.jsonl_path = self.metrics_dir / "metrics.jsonl"
        self.csv_path = self.metrics_dir / "metrics.csv"
        self._plot_every_n_logs = max(1, plot_every_n_logs)
        self._flush_on_log = flush_on_log
        self._logs_since_plot = 0
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._rows = self._read_existing_rows()
        if resume_step is not None:
            self._rows = [row for row in self._rows if int(row.get("step", -1)) < resume_step]
            self._rewrite_jsonl()
            self.flush()

        # Ensure metrics are written even if training is interrupted
        atexit.register(self._on_exit)

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        row: dict[str, int | float | str | bool] = {"step": int(step)}
        for key, value in metrics.items():
            scalar = _to_scalar(value)
            if scalar is not None:
                row[key] = scalar

        if len(row) == 1:
            return

        self._rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")

        self._write_csv()
        self._logs_since_plot += 1
        if self._flush_on_log or self._logs_since_plot >= self._plot_every_n_logs:
            self._write_plots()
            self._logs_since_plot = 0

    def flush(self) -> None:
        self._write_csv()
        self._write_plots()
        self._logs_since_plot = 0

    def _on_exit(self) -> None:
        """Ensure metrics are flushed even on SIGTERM/SIGINT."""
        try:
            self.flush()
        except Exception:
            pass

    def _read_existing_rows(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []

        rows = []
        with self.jsonl_path.open("r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _rewrite_jsonl(self) -> None:
        with self.jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for row in self._rows:
                jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_csv(self) -> None:
        if not self._rows:
            return

        metric_names = sorted({key for row in self._rows for key in row if key != "step"})
        fieldnames = ["step", *metric_names]
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

    def _write_plots(self) -> None:
        numeric_series = self._numeric_series()
        if not numeric_series:
            return

        try:
            self._write_matplotlib_plots(numeric_series)
        except (ImportError, ModuleNotFoundError):
            self._write_pillow_plots(numeric_series)

    def _numeric_series(self) -> dict[str, tuple[list[int], list[float]]]:
        metric_names = sorted({key for row in self._rows for key in row if key != "step"})
        series = {}
        for name in metric_names:
            steps = []
            values = []
            for row in self._rows:
                value = row.get(name)
                if isinstance(value, numbers.Real) and math.isfinite(float(value)):
                    steps.append(int(row["step"]))
                    values.append(float(value))
            if steps:
                series[name] = (steps, values)
        return series

    def _write_matplotlib_plots(self, series: dict[str, tuple[list[int], list[float]]]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(series)
        fig, axes = plt.subplots(len(names), 1, figsize=(11, max(2.4, 2.3 * len(names))), squeeze=False)
        for axis, name in zip(axes[:, 0], names, strict=True):
            steps, values = series[name]
            axis.plot(steps, values, linewidth=1.1, marker=".", markersize=2, label="raw")
            if len(values) >= 8:
                window = max(3, min(25, len(values) // 20))
                axis.plot(steps, _moving_average(values, window), linewidth=1.4, label=f"ma{window}")
                axis.legend(loc="best")
            axis.set_title(name)
            axis.set_xlabel("step")
            axis.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(self.plots_dir / "training_curves.png", dpi=180)
        plt.close(fig)

        for name, (steps, values) in series.items():
            fig, axis = plt.subplots(figsize=(11, 3.5))
            axis.plot(steps, values, linewidth=1.1, marker=".", markersize=2, label="raw")
            if len(values) >= 8:
                window = max(3, min(25, len(values) // 20))
                axis.plot(steps, _moving_average(values, window), linewidth=1.4, label=f"ma{window}")
                axis.legend(loc="best")
            axis.set_title(name)
            axis.set_xlabel("step")
            axis.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(self.plots_dir / f"{_safe_filename(name)}.png", dpi=180)
            plt.close(fig)

    def _write_pillow_plots(self, series: dict[str, tuple[list[int], list[float]]]) -> None:
        try:
            from PIL import Image
            from PIL import ImageDraw
        except ImportError:
            return

        names = list(series)
        panel_width = 1200
        panel_height = 280
        margin_left = 80
        margin_right = 30
        margin_top = 38
        margin_bottom = 44
        image = Image.new("RGB", (panel_width, panel_height * len(names)), "white")
        draw = ImageDraw.Draw(image)

        for panel_index, name in enumerate(names):
            y_offset = panel_index * panel_height
            self._draw_pillow_panel(
                draw,
                name,
                series[name],
                y_offset=y_offset,
                panel_width=panel_width,
                panel_height=panel_height,
                margin_left=margin_left,
                margin_right=margin_right,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
            )

        image.save(self.plots_dir / "training_curves.png")

        for name in names:
            image = Image.new("RGB", (panel_width, panel_height), "white")
            draw = ImageDraw.Draw(image)
            self._draw_pillow_panel(
                draw,
                name,
                series[name],
                y_offset=0,
                panel_width=panel_width,
                panel_height=panel_height,
                margin_left=margin_left,
                margin_right=margin_right,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
            )
            image.save(self.plots_dir / f"{_safe_filename(name)}.png")

    def _draw_pillow_panel(
        self,
        draw: Any,
        name: str,
        series: tuple[list[int], list[float]],
        *,
        y_offset: int,
        panel_width: int,
        panel_height: int,
        margin_left: int,
        margin_right: int,
        margin_top: int,
        margin_bottom: int,
    ) -> None:
        steps, values = series
        min_step = min(steps)
        max_step = max(steps)
        min_value = min(values)
        max_value = max(values)
        if min_step == max_step:
            max_step += 1
        if min_value == max_value:
            pad = abs(min_value) * 0.05 or 1.0
            min_value -= pad
            max_value += pad

        left = margin_left
        right = panel_width - margin_right
        top = y_offset + margin_top
        bottom = y_offset + panel_height - margin_bottom
        draw.rectangle((left, top, right, bottom), outline=(210, 210, 210), width=1)
        draw.text((left, y_offset + 10), name, fill=(20, 20, 20))
        draw.text((left, bottom + 12), f"step {min_step}", fill=(90, 90, 90))
        draw.text((right - 110, bottom + 12), f"step {max_step}", fill=(90, 90, 90))
        draw.text((10, top), f"{max_value:.4g}", fill=(90, 90, 90))
        draw.text((10, bottom - 12), f"{min_value:.4g}", fill=(90, 90, 90))

        points = []
        for step, value in zip(steps, values, strict=True):
            x_ratio = (step - min_step) / (max_step - min_step)
            y_ratio = (value - min_value) / (max_value - min_value)
            x = left + x_ratio * (right - left)
            y = bottom - y_ratio * (bottom - top)
            points.append((x, y))
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(30, 100, 200))
        else:
            draw.line(points, fill=(30, 100, 200), width=2)
