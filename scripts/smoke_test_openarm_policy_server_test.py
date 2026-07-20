import numpy as np
import pytest

from openpi.training import config as training_config
from scripts import smoke_test_openarm_policy_server as smoke


def test_openarm_smoke_observation_uses_16d_hq_contract() -> None:
    observation = smoke._observation(smoke.DEFAULT_PROMPT)  # noqa: SLF001

    assert observation["observation.state"].shape == (16,)
    assert observation["observation.images.base"].shape == (224, 224, 3)
    assert observation["prompt"] == smoke.DEFAULT_PROMPT


def test_action_summary_requires_finite_50_by_16_actions() -> None:
    summary = smoke._action_summary(np.zeros((50, 16), dtype=np.float32))  # noqa: SLF001
    assert summary["shape"] == [50, 16]

    with pytest.raises(ValueError, match="shape"):
        smoke._action_summary(np.zeros((50, 32), dtype=np.float32))  # noqa: SLF001
    invalid = np.zeros((50, 16), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        smoke._action_summary(invalid)  # noqa: SLF001


def test_contract_summary_requires_hq_units_and_dimensions() -> None:
    metadata = {
        "action_horizon": 50,
        "robot_action_dim": 16,
        "output_action_dim": 16,
        "action_unit": "degrees",
        "gripper_unit": "hq_motor_degrees",
        "gripper_open": 0.0,
        "gripper_closed": -66.0,
    }
    assert smoke._contract_summary(metadata)["passed"]  # noqa: SLF001

    metadata["action_unit"] = "radians"
    with pytest.raises(ValueError, match="metadata contract mismatch"):
        smoke._contract_summary(metadata)  # noqa: SLF001


def test_k_policy_config_publishes_openarm_server_contract() -> None:
    metadata = training_config.get_config("pi05_openarm_kai0_awbc_v1").policy_metadata

    assert metadata is not None
    metadata_with_horizon = {"action_horizon": 50, **metadata}
    assert smoke._contract_summary(metadata_with_horizon)["passed"]  # noqa: SLF001
