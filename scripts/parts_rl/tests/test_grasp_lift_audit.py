"""Evidence gate must not turn missing lift or user non-grasp into an RL zero."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("audit_grasp_lift", Path(__file__).parents[1] / "audit_grasp_lift.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sample(tmp_path, z):
    (tmp_path / "frame.jpg").touch()
    trace = {"time": list(range(len(z))), "arms": {"left": {"position_m": [[0, 0, x] for x in z]}}}
    candidate = {
        "candidate_id": "example", "episode_number": 1, "arm": "left", "split": "validation",
        "close_row": 1, "inspection_end_row": 3, "suggested_label": "failure",
        "label_source": "assistant_visual_audit",
        "frames": [{"row": r, "views": {"left": {
            "image": "frame.jpg", "video_frame_index": r, "capture_time": float(r)
        }}} for r in range(len(z))],
    }
    return candidate, trace


def test_preclosure_and_postrelease_height_cannot_qualify(tmp_path):
    candidate, trace = sample(tmp_path, [0.4, 0.02, 0.01, 0.02, 0.4])
    evidence = audit.lift_evidence(candidate, trace, tmp_path)
    assert not evidence["has_lift_evidence"]
    assert evidence["max_postclosure_rise_m"] == 0
    assert audit.proposal(candidate, {}, evidence)["proposed_reward"] is None


def test_height_does_not_imply_success_or_approved_reward(tmp_path):
    candidate, trace = sample(tmp_path, [0.4, 0.0, 0.06, 0.1])
    evidence = audit.lift_evidence(candidate, trace, tmp_path)
    assert evidence["has_lift_evidence"]
    result = audit.proposal(candidate, {}, evidence)
    assert result["proposed_reward"] == 0
    assert result["approved_reward"] is None
    assert not result["ready_for_training"]
    result = audit.proposal(candidate, {"review_label": "success"}, evidence)
    assert result["proposed_reward"] == 1


def test_user_non_grasp_is_preserved_even_with_lift(tmp_path):
    candidate, trace = sample(tmp_path, [0.4, 0.0, 0.06, 0.1])
    result = audit.proposal(candidate, {"review_label": "non_grasp", "notes": "调整"},
                            audit.lift_evidence(candidate, trace, tmp_path))
    assert result["effective_label"] == "non_grasp"
    assert result["notes"] == "调整"
    assert result["proposed_reward"] is None


def test_stale_wrist_image_cannot_qualify(tmp_path):
    candidate, trace = sample(tmp_path, [0.4, 0.0, 0.06, 0.1])
    for frame in candidate["frames"]:
        frame["views"]["left"]["capture_time"] -= 0.5
    assert not audit.lift_evidence(candidate, trace, tmp_path)["has_lift_evidence"]


def test_reviews_require_identity_and_unique_complete_ids():
    data = {"schema": "revised_grasp_human_review_v3", "identity": "wrong", "records": []}
    with pytest.raises(ValueError, match="identity"):
        audit.load_reviews(data, [], "correct")
    data.update(identity="correct", records=[{"candidate_id": "a"}, {"candidate_id": "a"}])
    with pytest.raises(ValueError, match="exactly once"):
        audit.load_reviews(data, [{"candidate_id": "a"}], "correct")
