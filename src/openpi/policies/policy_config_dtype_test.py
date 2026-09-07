import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies import policy
from openpi.policies import policy_config


@pytest.mark.parametrize(
    ("choice", "expected"), [("checkpoint", None), ("float32", jnp.float32), ("bfloat16", jnp.bfloat16)]
)
def test_jax_restore_dtype_is_explicit(tmp_path, monkeypatch, choice, expected):
    restore = Mock(return_value={"lora": "unchanged"})
    load = Mock()
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr(model, "restore_params", restore)
    monkeypatch.setattr(policy, "Policy", Mock())
    data = SimpleNamespace(
        data_transforms=transforms.Group(), model_transforms=transforms.Group(), use_quantile_norm=False
    )
    config = SimpleNamespace(
        model=SimpleNamespace(load=load),
        data=SimpleNamespace(create=lambda *_: data),
        assets_dirs=[],
        policy_metadata={},
    )
    policy_config.create_trained_policy(config, tmp_path, norm_stats={}, jax_param_dtype=choice)
    restore.assert_called_once_with(tmp_path / "params", dtype=expected)
    load.assert_called_once_with({"lora": "unchanged"})


def test_legacy_dtype_default_is_preserved():
    assert inspect.signature(policy_config.create_trained_policy).parameters["jax_param_dtype"].default == "bfloat16"
