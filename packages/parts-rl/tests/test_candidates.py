"""Fresh evidence and uncertain proposals; pure arrays, no neural/training code."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parts_rl.candidates import closing_events
from parts_rl.candidates import longest_run
from parts_rl.candidates import suggest_outcome


def trace(values, **kwargs):
    n = len(values)
    data = {"times": np.arange(n) * 0.05, "openness": values,
            "valid": np.ones(n, bool), "epochs": np.ones(n), "ticks": np.arange(n)}
    return closing_events(**(data | kwargs))


def test_one_closure_per_open_close_cycle_not_per_closed_feedback():
    events = trace([0.9, 0.8, 0.6] + [0.3] * 6 + [0.8, 0.9, 0.55])
    assert len(events) == 2
    assert events[0]["close"] == 2
    assert events[0]["end_reason"] == "reopened"
    assert events[1]["end_reason"] == "episode_end"


def test_epoch_break_ends_candidate_and_never_joins_across_gap():
    events = trace([0.9, 0.6, 0.3, 0.3, 0.2, 0.9, 0.6], epochs=[1, 1, 1, 2, 2, 2, 2])
    assert events[0]["end"] == 2
    assert events[0]["end_reason"] == "trace_gap"
    assert len(events) == 2
    assert events[1]["entry"] >= 3


def test_repeated_camera_frame_does_not_accumulate_hold_time():
    samples = [{"frame_key": "frozen", "visual_time": t} for t in np.arange(0, 2, 0.1)]
    assert longest_run(samples, lambda _: True) == 0


def test_gap_and_out_of_order_capture_cannot_create_longer_hold():
    samples = [{"frame_key": str(i), "visual_time": t} for i, t in enumerate([0, 0.2, 0.4, -1, 0.6, 0.8, 1.5])]
    assert longest_run(samples, lambda _: True) == pytest.approx(0.4)


def example(*, held=False, empty=False, lift=0.06, n=7):
    candidate = {"candidate_id": "E01-L001", "reference_base_z_m": 0.10, "end_reason": "reopened",
                 "samples": []}
    predictions = {}
    for i in range(n):
        t = i * 0.2
        key = f"frame-{i}"
        candidate["samples"].append({"frame_key": key, "visual_time": t, "time": t + 0.02,
                                      "valid": True, "base_z_m": 0.10 + lift, "openness": 0.4})
        predictions[key] = {"geometry": {"holding_candidate": held, "empty_candidate": empty,
                                          "held_objects": [{"score": 0.9, "area": 1000,
                                                            "centroid": [320, 420]}] if held else []}}
    return candidate, predictions


def test_positive_hold_requires_relative_lift_freshness_and_one_second():
    candidate, predictions = example(held=True)
    result = suggest_outcome(candidate, predictions)
    assert result["suggested_label"] == "success"
    assert result["visual_hold_s"] == pytest.approx(1.2)
    assert result["approved_reward"] is None
    assert result["review_label"] is None
    assert result["review_status"] == "unreviewed"


@pytest.mark.parametrize("reason", ["no_lift", "short", "frozen", "stale", "missing_time", "object_switch"])
def test_insufficient_hold_evidence_is_uncertain_not_failure(reason):
    candidate, predictions = example(held=True, lift=0.02 if reason == "no_lift" else 0.06,
                                      n=4 if reason == "short" else 7)
    for i, sample in enumerate(candidate["samples"]):
        if reason == "frozen":
            sample["frame_key"] = "frame-0"
            sample["visual_time"] = 0
        if reason == "stale":
            sample["time"] += 0.5
        if reason == "missing_time":
            sample["visual_time"] = None
        if reason == "object_switch":
            predictions[sample["frame_key"]]["geometry"]["held_objects"][0]["centroid"] = [100 if i % 2 else 500, 420]
    assert suggest_outcome(candidate, predictions)["suggested_label"] == "uncertain"


def test_zero_detection_is_not_failed_grasp_but_positive_empty_can_be_proposed():
    candidate, predictions = example()
    assert suggest_outcome(candidate, predictions)["suggested_label"] == "uncertain"
    candidate, predictions = example(empty=True)
    result = suggest_outcome(candidate, predictions)
    assert result["suggested_label"] == "failure"
    assert result["approved_reward"] is None


def test_short_hold_then_empty_is_not_relabelled_failure():
    candidate, predictions = example(empty=True)
    geometry = predictions["frame-0"]["geometry"]
    geometry.update(holding_candidate=True, held_objects=[{"score": 0.9, "area": 1000, "centroid": [320, 420]}])
    assert suggest_outcome(candidate, predictions)["suggested_label"] == "uncertain"
