import pathlib

import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import model as _model
import openpi.models.pi0_config as _pi0_config
import openpi.policies.yam_policy as _yam_policy
import openpi.training.config as _config


def _image() -> np.ndarray:
    return np.zeros((3, 8, 12), dtype=np.uint8)


def _observation() -> dict:
    return {
        "observation.images.top_rgb": _image(),
        "observation.images.left_rgb": _image(),
        "observation.images.right_rgb": _image(),
        "observation.state": np.arange(14, dtype=np.float32),
        "action": np.ones((4, 14), dtype=np.float32),
        "prompt": np.asarray("put the bottle into the bin"),
        "episode_index": np.asarray(2, dtype=np.int64),
    }


def test_yam_inputs_use_three_cameras_and_14d_bimanual_contract():
    output = _yam_policy.YamInputs(model_type=_model.ModelType.PI05)(_observation())

    assert output["state"].shape == (14,)
    assert output["actions"].shape == (4, 14)
    assert output["image"]["base_0_rgb"].shape == (8, 12, 3)
    assert set(output["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert output["episode_index"] == 2


def test_yam_inputs_reject_non_yam_dimensions():
    transform = _yam_policy.YamInputs(model_type=_model.ModelType.PI05)
    data = _observation()
    data["observation.state"] = np.zeros(16, dtype=np.float32)

    with pytest.raises(ValueError, match="YAM state must be 14D"):
        transform(data)

    data = _observation()
    data["action"] = np.zeros((4, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="YAM actions must be 14D"):
        transform(data)


def test_yam_outputs_remove_model_padding():
    transform = _yam_policy.YamOutputs()

    output = transform({"actions": np.zeros((50, 32), dtype=np.float32)})

    assert output["actions"].shape == (50, 14)

    with pytest.raises(ValueError, match="at least 14 dims"):
        transform({"actions": np.zeros((50, 7), dtype=np.float32)})


@pytest.mark.parametrize("key", ["observation.state", "action"])
def test_yam_inputs_reject_nonfinite(key):
    data = _observation()
    data[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        _yam_policy.YamInputs(model_type=_model.ModelType.PI05)(data)


def test_yam_data_config_uses_bimanual_action_mask_and_fixed_asset(tmp_path, monkeypatch):
    # The data-contract assertion does not need to download the PaliGemma tokenizer.
    monkeypatch.setattr(
        _config,
        "ModelTransformFactory",
        lambda **_: lambda _model_config: _transforms.Group(inputs=[]),
    )
    data = _config.LeRobotYamDataConfig(repo_id="local/test", base_config=_config.DataConfig(prompt_from_task=True))
    created = data.create(
        assets_dirs=pathlib.Path(tmp_path),
        model_config=_pi0_config.Pi0Config(pi05=True),
    )

    assert created.asset_id == "yam"
    assert created.lerobot_video_backend == "pyav"
    assert created.action_sequence_keys == ("action",)
    assert isinstance(created.data_transforms.inputs[1], _transforms.DeltaActions)
    assert created.data_transforms.inputs[1].mask == (True,) * 6 + (False,) + (True,) * 6 + (False,)
    assert isinstance(created.data_transforms.outputs[0], _transforms.AbsoluteActions)
    assert isinstance(created.data_transforms.outputs[1], _yam_policy.YamOutputs)
