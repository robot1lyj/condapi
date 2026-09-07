import hashlib

import numpy as np
import pytest

from scripts.thor.benchmark_pi05 import digest
from scripts.thor.benchmark_pi05 import read_observation


def observation():
    return {
        "observation.state": np.zeros(14, dtype=np.float32),
        "prompt": np.array("unit-test fixture only"),
        **{
            f"observation.images.{view}": np.zeros((16, 16, 3), dtype=np.uint8)
            for view in ("top_rgb", "left_rgb", "right_rgb")
        },
    }


def test_read_local_observation(tmp_path):
    path = tmp_path / "sample.npz"
    np.savez(path, **observation())
    result = read_observation(path)
    assert result["observation.state"].shape == (14,)
    assert result["prompt"] == "unit-test fixture only"
    assert digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("bad_state", [np.zeros(16), np.full(14, np.nan)])
def test_reject_invalid_state(tmp_path, bad_state):
    sample = observation()
    sample["observation.state"] = bad_state
    path = tmp_path / "sample.npz"
    np.savez(path, **sample)
    with pytest.raises(ValueError, match="finite 14D"):
        read_observation(path)


def test_reject_non_rgb_uint8(tmp_path):
    sample = observation()
    sample["observation.images.top_rgb"] = np.zeros((16, 16, 3), dtype=np.float32)
    path = tmp_path / "sample.npz"
    np.savez(path, **sample)
    with pytest.raises(ValueError, match="uint8 RGB"):
        read_observation(path)
