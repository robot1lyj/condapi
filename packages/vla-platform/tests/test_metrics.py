import json
import logging

import pytest
from vla_platform.metrics import write_metrics

from adapters.lerobot.metrics import TrackerMetrics


def test_writer_and_tracker(tmp_path):
    path = tmp_path / "run.metrics/metrics.jsonl"
    write_metrics(path, 1, {"custom/loss": 0.123456789})
    tracker_type = type(
        "MetricsTracker",
        (),
        {
            "__module__": "lerobot.utils.logging_utils",
            "to_dict": lambda self: {"steps": 2, "loss": 0.123456789, "step_s": 0.5, "samples_per_s": 8},
        },
    )
    capture = TrackerMetrics(path)
    record = logging.LogRecord("root", logging.INFO, "test", 1, tracker_type(), (), None)
    assert capture.filter(record)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]["metrics"]["loss"] == 0.123456789
    assert rows[1]["metrics"]["samples_per_second"] == 8
    assert rows[1]["step"] == 2


@pytest.mark.parametrize("metrics", [{"loss": float("nan")}, {"loss": True}, {"loss": "1"}])
def test_writer_rejects_bad_values(tmp_path, metrics):
    with pytest.raises(ValueError, match="finite scalars"):
        write_metrics(tmp_path / "metrics.jsonl", 1, metrics)
    assert not (tmp_path / "metrics.jsonl").exists()
