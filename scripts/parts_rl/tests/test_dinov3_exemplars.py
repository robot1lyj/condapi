"""Perception evaluation isolation and uncertainty checks; no neural inference."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("dino_eval", Path(__file__).parents[1] / "evaluate_dinov3.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def fixture():
    jobs = [
        {"id": name, "sha256": name, "episode_number": ep, "split": split, "roi_measurement": {"geometry_valid": True}}
        for name, ep, split in [("held", 1, "reference"), ("empty", 2, "reference"), ("query", 3, "evaluation")]
    ]
    features = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    return jobs, features, {"held": "held_at_snapshot", "empty": "empty_at_snapshot"}


def test_same_episode_cannot_be_both_reference_and_evaluation():
    jobs, features, labels = fixture()
    jobs[-1]["episode_number"] = jobs[0]["episode_number"]
    with pytest.raises(ValueError, match="episodes overlap"):
        evaluation.predict(features, jobs, labels)


def test_same_image_cannot_leak_across_splits_under_a_new_id():
    jobs, features, labels = fixture()
    jobs[-1]["sha256"] = jobs[0]["sha256"]
    with pytest.raises(ValueError, match="images overlap"):
        evaluation.predict(features, jobs, labels)


def test_evaluation_label_cannot_enter_prediction():
    jobs, features, labels = fixture()
    labels["query"] = "held_at_snapshot"
    with pytest.raises(ValueError, match="reference labels"):
        evaluation.predict(features, jobs, labels)


def test_bad_geometry_stays_unknown_despite_held_feature_match():
    jobs, features, labels = fixture()
    assert evaluation.predict(features, jobs, labels)[0]["label"] == "held_at_snapshot"
    jobs[-1]["roi_measurement"]["geometry_valid"] = False
    assert evaluation.predict(features, jobs, labels)[0]["label"] == "uncertain"


def test_ambiguous_feature_and_ambiguous_audit_are_not_failure_labels():
    jobs, features, labels = fixture()
    features[-1] = [1.0, 1.0]
    pred = evaluation.predict(features, jobs, labels)
    assert pred[0]["label"] == "uncertain"
    counts = evaluation.metrics(pred, {"query": "uncertain"})
    assert counts["audit_uncertain"] == 1
    assert counts["fn"] == counts["tn"] == 0
