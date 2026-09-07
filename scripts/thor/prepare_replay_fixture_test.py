import numpy as np

from scripts.thor.prepare_replay_fixture import action_chunks


def test_delta_horizon_uses_current_state_and_preserves_grippers():
    state = np.arange(3 * 14, dtype=np.float32).reshape(3, 14)
    action = state + 2
    result = action_chunks(state, action, 1, 3, horizon=3)["actions"]
    np.testing.assert_array_equal(result[0, :, 0], [2, 16, 16])
    np.testing.assert_array_equal(result[1, :, 0], [2, 2, 2])
    np.testing.assert_array_equal(result[0, :, 6], action[[1, 2, 2], 6])
    np.testing.assert_array_equal(result[0, :, 13], action[[1, 2, 2], 13])
    # Source arrays must never be changed by calibration.
    np.testing.assert_array_equal(action, state + 2)
