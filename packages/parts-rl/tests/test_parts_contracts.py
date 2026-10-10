"""Pure array/supervision checks. Never import Torch or execute training."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parts_rl.actions import ActionContract
from parts_rl.actions import explore
from parts_rl.actions import project_chunk
from parts_rl.supervision import Evidence
from parts_rl.supervision import GraspContract
from parts_rl.supervision import GraspSupervisor
from parts_rl.supervision import reset_intent


def contract():
    return GraspContract("right", "reviewed-test-frame", 0.01, 0.05, 0.8, 0.6, 3.0)


def evidence(tick, **kwargs):
    return Evidence(**{
        "epoch": 1, "tick": tick, "time_s": tick / 10,
        "observation_id": f"obs-{tick}", "visual_time_s": tick / 10,
        "visual_id": f"image-{tick}", "calibration_id": "reviewed-test-frame",
        "height_m": 0.03, "openness": 0.9, "pose_valid": True,
        "centered_brick": True, "outside_bins": True,
        "holding_brick": False, "empty_hand": True, **kwargs,
    })


def active(mode="evaluate"):
    supervisor = GraspSupervisor(contract(), mode=mode)
    events = []
    for tick in range(3):
        events.extend(supervisor.update(evidence(tick)))
    assert supervisor.phase == "active"
    assert len(events) == 1
    assert events[0]["type"] == "attempt_start"
    return supervisor


def held(tick, **kwargs):
    return evidence(tick, height_m=0.09, openness=0.4, holding_brick=True, empty_hand=False, **kwargs)


@pytest.mark.parametrize(("arm", "index"), [("left", 6), ("right", 13)])
def test_only_one_gripper_changes_and_committed_prefix_is_exact(arm, index):
    base = np.full((50, 14), 0.6, dtype=np.float64)
    editable = np.arange(50) >= 9
    action = ActionContract(arm, "pi20w", "feature", "openness")
    target, applied = project_chunk(base, np.full((50, 1), -1.0), editable, action, committed_prefix=base[:9])
    np.testing.assert_array_equal(target[:9], base[:9])
    np.testing.assert_array_equal(target[:, np.arange(14) != index], base[:, np.arange(14) != index])
    np.testing.assert_allclose(target[9:, index], 0.1)
    np.testing.assert_array_equal(applied[:9], 0)


def test_replay_residual_reflects_clipping_not_unexecuted_proposal():
    base = np.full((50, 14), 0.1)
    action = ActionContract("right", "pi20w", "feature", "openness")
    target, applied = project_chunk(base, -np.ones((50, 1)), np.ones(50, bool), action,
                                    committed_prefix=np.empty((0, 14)))
    np.testing.assert_array_equal(target[:, 13], 0)
    np.testing.assert_allclose(applied, -0.2)


def test_smoothing_continues_after_the_commitment_without_editing_it():
    base = np.full((50, 14), 0.5)
    action = ActionContract("right", "pi20w", "feature", "openness", smoothing_weight=0.5)
    target, applied = project_chunk(base, np.zeros((50, 1)), np.arange(50) >= 9, action,
                                    committed_prefix=base[:9], previous_u=-0.8)
    np.testing.assert_array_equal(target[:9], base[:9])
    assert applied[9, 0] == pytest.approx(-0.4)
    assert applied[10, 0] == pytest.approx(-0.2)


def test_prefix_mismatch_and_editable_commitments_are_rejected():
    base = np.full((50, 14), 0.5)
    action = ActionContract("right", "pi20w", "feature", "openness")
    with pytest.raises(ValueError, match="RTC"):
        project_chunk(base, np.zeros((50, 1)), np.ones(50, bool), action, committed_prefix=base[:2])
    mask = np.arange(50) >= 2
    with pytest.raises(ValueError, match="RTC"):
        project_chunk(base, np.zeros((50, 1)), mask, action, committed_prefix=base[:2] + 0.001)


def test_zero_residual_equals_pi_and_exploration_obeys_mask():
    mean = np.zeros((50, 1))
    mask = np.arange(50) >= 9
    np.testing.assert_array_equal(explore(mean, mask, std=0, rng=np.random.default_rng(0)), mean)
    noise = explore(mean, mask, std=2, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(noise[:9], 0)
    assert np.max(np.abs(noise)) <= 1


def test_duplicate_image_cannot_start_an_attempt():
    supervisor = GraspSupervisor(contract(), mode="evaluate")
    for tick in range(3):
        supervisor.update(evidence(tick, visual_id="same-image", visual_time_s=0))
    assert supervisor.phase == "base"


def test_lift_without_visual_hold_never_succeeds():
    supervisor = active()
    events = []
    for tick in range(3, 15):
        events.extend(supervisor.update(evidence(tick, height_m=0.2, openness=0.4)))
    assert not events
    assert supervisor.phase == "active"


def test_visual_hold_without_five_centimetre_lift_never_succeeds():
    supervisor = active()
    events = []
    for tick in range(3, 15):
        events.extend(supervisor.update(replace(held(tick), height_m=0.07)))
    assert not events
    assert supervisor.phase == "active"


def test_one_success_event_after_persistent_hold_then_handoff_without_reentry():
    supervisor = active()
    events = []
    for tick in range(3, 15):
        events.extend(supervisor.update(held(tick)))
    assert len(events) == 1
    assert events[0]["reward"] == 1
    assert events[0]["next"] == "handoff_to_pi"
    assert supervisor.phase == "carrying"
    for tick in range(15, 20):
        assert not supervisor.update(held(tick))


def test_a_gap_in_visual_evidence_restarts_the_hold_clock():
    supervisor = active()
    for tick in range(3, 9):
        assert not supervisor.update(held(tick))
    for tick in range(12, 20):
        assert not supervisor.update(held(tick))
    assert supervisor.phase == "active"


def test_unknown_timeout_is_not_a_failure_label():
    supervisor = active()
    events = []
    for tick in range(3, 33):
        events.extend(supervisor.update(evidence(tick, holding_brick=None, empty_hand=None)))
    assert len(events) == 1
    assert events[0]["reward"] is None
    assert supervisor.phase == "review"


def test_reopen_empty_is_failure_and_epoch_change_is_unknown():
    supervisor = active()
    supervisor.update(evidence(3, openness=0.4))
    assert supervisor.update(evidence(4))[0]["reward"] == 0
    supervisor = active()
    assert supervisor.update(evidence(3, epoch=2))[0]["reward"] is None


def test_practice_requests_reset_but_evaluation_cannot_reset():
    supervisor = active("practice")
    events = []
    for tick in range(3, 15):
        events.extend(supervisor.update(held(tick)))
    assert supervisor.phase == "reset_pending"
    intent = reset_intent(events[0], entry_height_m=0.03, budget_s=5)
    assert intent["executes_motion"] is False
    with pytest.raises(ValueError, match="reset"):
        reset_intent({**events[0], "next": "handoff_to_pi"}, entry_height_m=0.03, budget_s=5)
    with pytest.raises(ValueError, match="empty hand"):
        supervisor.reset_acknowledged(held(15))
    supervisor.reset_acknowledged(evidence(15))
    assert supervisor.phase == "base"


def test_invalid_pose_calibration_and_out_of_order_feedback_cannot_label_success():
    supervisor = active()
    events = []
    for tick in range(3, 15):
        events.extend(supervisor.update(held(tick, calibration_id="other-frame")))
    assert not events
    with pytest.raises(ValueError, match="duplicate"):
        supervisor.update(evidence(14))


def test_protocol_checks_do_not_import_torch():
    assert "torch" not in sys.modules
