"""Physical identity must survive missing images; camera evidence stays distinct."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "revise_grasp_review", Path(__file__).parents[1] / "revise_grasp_review.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

vlm_spec = importlib.util.spec_from_file_location(
    "review_grasps_vlm", Path(__file__).parents[1] / "review_grasps_vlm.py"
)
vlm = importlib.util.module_from_spec(vlm_spec)
vlm_spec.loader.exec_module(vlm)


def test_camera_gap_does_not_end_physical_attempt():
    t = np.arange(60) / 30
    g = np.r_[np.ones(5), np.linspace(1, 0.1, 10), np.full(30, 0.1), np.linspace(0.1, 1, 15)]
    ev = review.physical_events(t, g, np.zeros(60), np.ones(60, bool), np.zeros(60, int), np.arange(60))
    assert len(ev) == 1
    assert ev[0]["end_reason"] == "reopened"
    camera_valid = np.ones(60, bool)
    camera_valid[16:22] = False
    rows = review.select_rows(t, ev[0]["close"], ev[0]["end"], camera_valid, np.arange(60) / 100)
    assert rows
    assert not any(16 <= r < 22 for r in rows)
    assert any(r > 22 for r in rows)


def test_real_epoch_break_still_ends_attempt():
    t = np.arange(50) / 30
    g = np.r_[np.ones(5), np.linspace(1, 0.1, 10), np.full(35, 0.1)]
    epochs = np.r_[np.zeros(20, int), np.ones(30, int)]
    ev = review.physical_events(t, g, np.zeros(50), np.ones(50, bool), epochs, np.arange(50))
    assert ev[0]["end"] == 19
    assert ev[0]["end_reason"] == "trace_gap"


def test_slow_closure_and_long_hold_are_not_cut_by_camera_window():
    t = np.arange(420) / 30
    g = np.r_[np.ones(10), np.linspace(1, 0.1, 90), np.full(300, 0.1), np.linspace(0.1, 1, 20)]
    ev = review.physical_events(t, g, np.zeros(420), np.ones(420, bool), np.zeros(420, int), np.arange(420))
    assert len(ev) == 1
    assert ev[0]["end_reason"] == "reopened"
    assert t[ev[0]["end"]] - t[ev[0]["close"]] > 8


def test_partial_open_retries_are_separate_attempts():
    g = np.r_[
        np.ones(5),
        np.linspace(1, 0.2, 10),
        np.full(10, 0.2),
        np.linspace(0.2, 0.68, 10),
        np.linspace(0.68, 0.2, 10),
        np.full(10, 0.2),
    ]
    n = len(g)
    events = review.physical_events(np.arange(n) / 30, g, np.zeros(n), np.ones(n, bool), np.zeros(n, int), np.arange(n))
    assert len(events) == 2
    assert events[0]["end_reason"] == "reopened"
    assert events[0]["end"] < events[1]["close"]


def test_resume_rejects_stale_prompt_or_invalid_evidence(tmp_path):
    candidate = {"candidate_id": "E01-L001-v3", "montage_sha256": "abc"}
    path = tmp_path / "decision.json"
    assert vlm.needs_inference(candidate, path, "prompt")
    saved = dict(
        candidate,
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        decision={"outcome": "success"},
        parse_error=None,
    )
    path.write_text(json.dumps(saved))
    assert not vlm.needs_inference(candidate, path, "prompt")
    with pytest.raises(ValueError, match="stale or invalid"):
        vlm.needs_inference(candidate, path, "different prompt")
    saved["parse_error"] = "truncated JSON"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="stale or invalid"):
        vlm.needs_inference(candidate, path, "prompt")
