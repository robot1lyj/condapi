"""Small tensor/config checks only: no model training or optimizer steps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import rtc_training


def test_clean_prefix_and_postfix_scale():
    actions = jnp.ones((2, 5, 3))
    noise = jnp.zeros_like(actions)
    x, time, prefix = rtc_training.condition_prefix(actions, noise, jnp.array([0.4, 0.8]), jnp.array([0, 4]))
    np.testing.assert_allclose(x[0], 0.6)
    np.testing.assert_allclose(x[1, :4], 1)
    np.testing.assert_array_equal(time[1, :4], 0)
    np.testing.assert_allclose(x[1, 4], 0.2, atol=1e-7)
    loss = rtc_training.postfix_loss(jnp.ones((2, 5)), prefix)
    np.testing.assert_allclose(loss.mean(axis=-1), [1, 1])
    np.testing.assert_array_equal(loss[1, :4], 0)


def test_zero_delay_preserves_inputs_and_loss():
    actions = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3)
    noise = -actions
    time = jnp.array([0.1, 0.9])
    x, _, prefix = rtc_training.condition_prefix(actions, noise, time, jnp.zeros(2, dtype=jnp.int32))
    expected = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    np.testing.assert_array_equal(x, expected)
    loss = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)
    np.testing.assert_array_equal(rtc_training.postfix_loss(loss, prefix), loss)
    assert rtc_training.postfix_loss(loss, None) is loss


def test_per_token_time_embedding_matches_scalar_embedding():
    time = jnp.array([0.2, 0.8])
    scalar = pi0.posemb_sincos(time, 8, 0.004, 4.0)
    tokens = pi0.posemb_sincos(jnp.repeat(time[:, None], 5, axis=1), 8, 0.004, 4.0)
    np.testing.assert_array_equal(tokens, jnp.repeat(scalar[:, None], 5, axis=1))


@pytest.mark.parametrize("delay", [-1, 50, 51, 1.5, True])
def test_invalid_delay(delay):
    with pytest.raises(ValueError, match="action_horizon"):
        pi0_config.Pi0Config(pi05=True, rtc_training_max_delay=delay)


def test_pi0_rejected_and_pi05_supported():
    with pytest.raises(ValueError, match=r"Pi0\.5"):
        pi0_config.Pi0Config(rtc_training_max_delay=10)
    assert pi0_config.Pi0Config(pi05=True, rtc_training_max_delay=10).action_horizon == 50


def test_adaptive_norm_reuses_weights_for_per_token_condition():
    norm = gemma.RMSNorm()
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    cond = jnp.ones((2, 4))
    variables = norm.init(jax.random.key(0), x, cond)
    # Nonzero modulation exercises scale/shift/gate, not just zero-init identity.
    variables["params"]["Dense_0"]["kernel"] = jnp.ones((4, 12)) * 0.1
    shared, shared_gate = norm.apply(variables, x, cond)
    per_token, gate = norm.apply(variables, x, jnp.repeat(cond[:, None], 3, axis=1))
    np.testing.assert_allclose(per_token, shared)
    np.testing.assert_allclose(gate, jnp.repeat(shared_gate, 3, axis=1))
    varied = jnp.repeat(cond[:, None], 3, axis=1).at[:, 1].set(0)
    changed, _ = norm.apply(variables, x, varied)
    np.testing.assert_allclose(changed[:, 0], shared[:, 0])
    assert not np.allclose(changed[:, 1], shared[:, 1])


def test_full_pi05_parameter_and_loss_shapes_only():
    # Abstract tracing only: no allocation of full weights, optimizer or training step.
    import flax.nnx as nnx  # noqa: PLC0415

    configs = [pi0_config.Pi0Config(pi05=True, rtc_training_max_delay=d) for d in (0, 10)]
    models = [nnx.eval_shape(config.create, jax.random.key(0)) for config in configs]
    states = [nnx.state(model, nnx.Param).flat_state() for model in models]
    assert set(states[0]) == set(states[1])
    for key in states[0]:
        assert states[0][key].value.shape == states[1][key].value.shape
        assert states[0][key].value.dtype == states[1][key].value.dtype
    obs, actions = configs[1].inputs_spec(batch_size=2)
    loss = nnx.eval_shape(
        lambda model, obs, actions: model.compute_loss(jax.random.key(1), obs, actions, train=False),
        models[1],
        obs,
        actions,
    )
    assert loss.shape == (2, 50)
