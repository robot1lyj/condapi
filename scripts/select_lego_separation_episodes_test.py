"""Pose-event regression checks; no robot, dataset, model or training loop."""

import numpy as np

from scripts.select_lego_separation_episodes import find_bimanual_events
from scripts.select_lego_separation_episodes import find_events


def trajectory():
    n = 90
    frames = np.arange(n) * 3
    positions = np.zeros((n, 3))
    positions[30:50, 2] = 0.08
    wrist = positions.copy()
    axes = np.zeros((n, 3))
    axes[:, 2] = -1
    axes[30:50] = [1, 0, 0]
    gripper = np.full(n, 0.2)
    return positions, wrist, axes, gripper, frames


def test_lift_and_horizontal_is_detected_with_source_frames():
    events = find_events(*trajectory(), 30, 3)
    assert len(events) == 1
    assert events[0]["start_frame"] == 90
    assert events[0]["end_frame_exclusive"] == 150
    assert events[0]["separation_success"] is None


def test_roll_or_lift_alone_does_not_trigger():
    data = list(trajectory())
    data[2][:] = [0, 0, -1]
    assert not find_events(*data, 30, 3)


def test_transient_horizontal_does_not_trigger():
    data = list(trajectory())
    data[2][31:50] = [0, 0, -1]
    assert not find_events(*data, 30, 3)


def test_home_pose_without_downward_history_does_not_trigger():
    data = list(trajectory())
    data[2][:] = [1, 0, 0]
    assert not find_events(*data, 30, 3)


def test_rotation_induced_tip_rise_is_exposed_and_open_gripper_not_rejected():
    data = list(trajectory())
    data[1][:] = 0
    data[3][:] = 0.95
    event = find_events(*data, 30, 3)[0]
    assert event["nominal_wrist_rise_m"] == 0
    assert event["closed_fraction_below_0_6"] == 0


def test_bimanual_requires_temporal_overlap_and_lift_history():
    positions, wrist, axes, grip, frames = trajectory()
    poses = np.zeros((len(frames), 2, 3, 3))
    poses[:, :, 0] = positions[:, None]
    poses[:, :, 1] = wrist[:, None]
    poses[:, :, 2] = axes[:, None]
    state = np.zeros((len(frames), 14))
    state[:, [6, 13]] = grip[:, None]
    unilateral = [{**e, "arm": "left"} for e in find_events(positions, wrist, axes, grip, frames, 30, 3)]
    events = find_bimanual_events(poses, state, frames, 30, 3, unilateral)
    assert len(events) == 1
    assert events[0]["arm"] == "both"
    assert events[0]["both_closed_fraction_below_0_6"] == 1
    assert not find_bimanual_events(poses, state, frames, 30, 3, [])
    poses[:, 1, 2] = [0, 0, -1]
    poses[55:75, 1, 2] = [1, 0, 0]
    assert not find_bimanual_events(poses, state, frames, 30, 3, unilateral)
