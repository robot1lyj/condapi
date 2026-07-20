import numpy as np
import pytest

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
