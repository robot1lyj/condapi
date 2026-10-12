import numpy as np
from yam_pld_rl.data import action_coordinates
from yam_pld_rl.data import fit_normalization
from yam_pld_rl.data import observed_edges
from yam_pld_rl.data import physical_edges
from yam_pld_rl.data import suffix_return


def test_camera_gap_drops_edges_but_physical_gap_breaks_mc():
    times = np.arange(6, dtype=float) / 30
    edges = physical_edges(times, np.arange(6), np.ones(6), np.ones(6, bool))
    valid = np.array([1, 1, 0, 1, 1, 1], bool)
    assert observed_edges(edges, valid).tolist() == [True, False, False, True, True]
    assert np.isfinite(suffix_return(times, edges, 5, 1, 0.99)[:5]).all()
    edges[2] = False
    result = suffix_return(times, edges, 5, 1, 0.99)
    assert np.isnan(result[:3]).all()
    assert result[4] == 1


def test_delta_is_only_joint_coordinates_and_is_copy():
    action = np.arange(14, dtype=np.float32)
    converted = action_coordinates(action, np.ones(14, np.float32))
    assert converted[6] == action[6]
    assert converted[13] == action[13]
    assert converted[0] == -1
    assert converted[7] == 6
    assert action[0] == 0


def test_validation_does_not_change_normalization():
    values = np.array([[0.0, 1.0], [2.0, 3.0], [1000.0, 1000.0]])
    stats = fit_normalization(values, values, np.array([1, 1, 0], bool))
    assert stats["state_mean"] == [1.0, 2.0]
