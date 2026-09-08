import numpy as np
import pytest

from scripts.compute_yam_norm_stats import action_chunks


def test_delta_chunk_keeps_grippers_absolute_and_repeats_episode_tail():
    states = np.arange(42, dtype=np.float32).reshape(3, 14)
    actions = states + 2
    result = action_chunks(states, actions, np.array([0, 2]), 50)
    for batch_index, frame in enumerate([0, 2]):
        for offset in range(50):
            expected = actions[min(frame + offset, 2)].copy()
            for dim in range(14):
                if dim not in (6, 13):
                    expected[dim] -= states[frame, dim]
            np.testing.assert_array_equal(result["actions"][batch_index, offset], expected)
    np.testing.assert_array_equal(actions, states + 2)


def test_single_frame_episode_is_valid():
    result = action_chunks(np.zeros((1, 14)), np.ones((1, 14)), np.array([0]), 50)
    np.testing.assert_array_equal(result["actions"], np.ones((1, 50, 14)))


@pytest.mark.parametrize("shape", [(3, 16), (3, 7), (14,)])
def test_reject_wrong_contract(shape):
    with pytest.raises(ValueError, match="Nx14"):
        action_chunks(np.zeros(shape), np.zeros(shape), np.array([0]), 50)


def test_reject_nonfinite():
    actions = np.zeros((1, 14))
    actions[0, 6] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        action_chunks(np.zeros((1, 14)), actions, np.array([0]), 50)
