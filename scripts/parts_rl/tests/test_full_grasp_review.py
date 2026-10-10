"""Check evidence isolation and temporal failures without model execution."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
SPEC = importlib.util.spec_from_file_location("full_review", Path(__file__).parents[1] / "classify_full_grasp_review.py")
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


def frame(label, time, sequence, identity):
    return {"label": label, "visual_time": time, "camera_sequence": sequence, "sha256": identity, "epoch": 0}


def test_same_episode_reference_cannot_vote_for_query():
    refs = [{"id": name, "episode_number": ep, "sha256": name, "crop": name,
             "roi_measurement": {"geometry_valid": True}}
            for name, ep in [("held_same", 1), ("held_other", 2), ("empty_other", 3)]]
    labels = {"held_same": "held_at_snapshot", "held_other": "held_at_snapshot", "empty_other": "empty_at_snapshot"}
    vectors = np.array([[1., 0.], [0., 1.], [.8, .6]])
    p = full.snapshot_prediction({"episode_number": 1, "sha256": "query"}, np.array([1., 0.]), refs, vectors, labels)
    assert p["label"] == "empty_at_snapshot"
    assert p["nearest"]["held_at_snapshot"]["id"] == "held_other"


def test_same_image_different_episode_cannot_vote_and_missing_class_stays_unknown():
    refs = [{"id": "held", "episode_number": 2, "sha256": "query", "roi_measurement": {"geometry_valid": True}},
            {"id": "empty", "episode_number": 3, "sha256": "other", "roi_measurement": {"geometry_valid": True}}]
    result = full.snapshot_prediction({"episode_number": 1, "sha256": "query"}, np.array([1., 0.]),
                                      refs, np.eye(2), {"held": "held_at_snapshot", "empty": "empty_at_snapshot"})
    assert result["label"] == "uncertain"


def test_missing_lifted_images_are_not_negative_reward():
    assert full.aggregate([])[0] == "uncertain"


def test_single_held_frame_is_not_multiframe_evidence():
    assert full.aggregate([frame("held_at_snapshot", 1., 1, "a")])[0] == "uncertain"


def test_repeated_capture_and_time_gap_cannot_accumulate_duration():
    for sequence, identity, time in [(1, "b", 1.2), (2, "a", 1.2), (2, "b", 1.5)]:
        evidence = [frame("held_at_snapshot", 1., 1, "a"), frame("held_at_snapshot", time, sequence, identity)]
        assert full.aggregate(evidence)[0] == "uncertain"


def test_unknown_breaks_an_apparent_held_run():
    evidence = [frame("held_at_snapshot", 1., 1, "a"), frame("uncertain", 1.1, 2, "b"),
                frame("held_at_snapshot", 1.2, 3, "c")]
    assert full.aggregate(evidence)[0] == "uncertain"


def test_held_then_empty_requires_review_instead_of_failure():
    evidence = [frame("held_at_snapshot", 1., 1, "a"), frame("held_at_snapshot", 1.2, 2, "b"),
                frame("empty_at_snapshot", 1.4, 3, "c")]
    assert full.aggregate(evidence)[0] == "uncertain"
    assert full.aggregate(evidence)[3] == "mixed_held_and_empty"


def test_clear_consistent_evidence_remains_a_proposal():
    for label, suggestion in [("held_at_snapshot", "held_observed"), ("empty_at_snapshot", "empty_observed")]:
        evidence = [frame(label, 1., 1, "a"), frame(label, 1.2, 2, "b")]
        assert full.aggregate(evidence)[0] == suggestion
