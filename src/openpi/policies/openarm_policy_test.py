import numpy as np
import pytest

from openpi.models import model as _model
import openpi.policies.openarm_policy as _openarm_policy


def _image() -> np.ndarray:
    return np.zeros((32, 48, 3), dtype=np.uint8)


def test_openarm_inputs_preserve_16d_contract_and_metadata():
    transform = _openarm_policy.OpenArmInputs(model_type=_model.ModelType.PI05)
    data = {
        "observation.images.base": _image(),
        "observation.images.left_wrist": _image(),
        "observation.images.right_wrist": _image(),
        "observation.state": np.arange(16, dtype=np.float32),
        "action": np.ones((4, 16), dtype=np.float32),
        "prompt": np.asarray("Fold the T-shirt properly"),
        "episode_index": np.asarray(3),
        "complementary_info.acp_indicator": np.asarray(1, dtype=np.int64),
    }

    output = transform(data)

    assert output["state"].shape == (16,)
    assert output["actions"].shape == (4, 16)
    assert set(output["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert output["episode_index"] == data["episode_index"]
    assert output["complementary_info.acp_indicator"] == data["complementary_info.acp_indicator"]


def test_openarm_inputs_reject_wrong_state_action_dims():
    transform = _openarm_policy.OpenArmInputs(model_type=_model.ModelType.PI05)
    data = {
        "observation.images.base": _image(),
        "observation.images.left_wrist": _image(),
        "observation.images.right_wrist": _image(),
        "observation.state": np.zeros(14, dtype=np.float32),
        "action": np.ones((4, 16), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="OpenArm state must be 16D"):
        transform(data)

    data["observation.state"] = np.zeros(16, dtype=np.float32)
    data["action"] = np.ones((4, 14), dtype=np.float32)
    with pytest.raises(ValueError, match="OpenArm actions must be 16D"):
        transform(data)


def test_openarm_outputs_slice_model_actions_to_16d():
    transform = _openarm_policy.OpenArmOutputs()

    output = transform({"actions": np.zeros((50, 32), dtype=np.float32)})

    assert output["actions"].shape == (50, 16)

    with pytest.raises(ValueError, match="at least 16 dims"):
        transform({"actions": np.zeros((50, 14), dtype=np.float32)})
