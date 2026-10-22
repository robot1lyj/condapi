"""Protocol tests with stubbed device operations; never executes a training update."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openpi.training import numerical_diagnostics as diagnostics


def stats(*, finite=True):
    return {
        "summary": {"all_finite": finite, "stable_finite_values_l2": 0.1, "bad_record_indexes": [] if finite else [0]}
    }


def harness(monkeypatch, inspections):
    agent = object.__new__(diagnostics.NumericalDiagnostics)
    agent.failure_written = False
    agent.context = {"pre_update_step": 0, "attempted_completed_step": 1}
    agent.inspect = Mock(side_effect=inspections)
    agent.record = Mock()

    def save_failure(*args, **kwargs):
        agent.failure_written = True

    agent.failure = Mock(side_effect=save_failure)
    monkeypatch.setattr(diagnostics.jax, "device_get", lambda value: value)
    return agent


def test_bad_input_never_calls_gradient_or_optimizer(monkeypatch):
    agent = harness(monkeypatch, [stats(finite=False)])
    gradient, optimizer = Mock(), Mock()
    state = SimpleNamespace(step=0)
    with pytest.raises(FloatingPointError, match="Nonfinite input"):
        agent.update(state, {"input": "actual"}, "rng", gradient, optimizer)
    gradient.assert_not_called()
    optimizer.assert_not_called()
    assert agent.failure.call_args.kwargs["state"] is state


def test_bad_gradient_saves_actual_pre_update_state_without_calling_optimizer(monkeypatch):
    agent = harness(monkeypatch, [stats(), stats(finite=False)])
    actual_gradient = {"weight": np.array([np.inf], np.float32)}
    gradient = Mock(return_value=(np.float32(0.01), actual_gradient, np.float32(np.inf)))
    optimizer = Mock()
    state = SimpleNamespace(step=0)
    with pytest.raises(FloatingPointError, match="BEFORE optimizer"):
        agent.update(state, "batch", "rng", gradient, optimizer)
    optimizer.assert_not_called()
    assert agent.failure.call_args.kwargs["state"] is state
    assert agent.failure.call_args.kwargs["gradients"] is actual_gradient


def test_post_update_failure_is_explicitly_labelled_and_preserves_actual_update(monkeypatch):
    agent = harness(monkeypatch, [stats(), stats(), stats(finite=False), stats(finite=False)])
    gradient = Mock(return_value=(0.01, {"weight": "gradient"}, 0.1))
    post_state = SimpleNamespace(params="post_params", opt_state="post_optimizer", step=1)
    updates = {"weight": "actual_update"}
    optimizer = Mock(return_value=(post_state, 1.0, updates))
    with pytest.raises(FloatingPointError, match="after update"):
        agent.update(SimpleNamespace(step=0), "batch", "rng", gradient, optimizer)
    optimizer.assert_called_once()
    assert agent.failure.call_args.kwargs["phase"] == "post_update"
    assert agent.failure.call_args.kwargs["state"] is post_state
    assert agent.failure.call_args.kwargs["updates"] is updates


def test_healthy_protocol_calls_each_native_phase_once(monkeypatch):
    agent = harness(monkeypatch, [stats(), stats(), stats(), stats()])
    gradient = Mock(return_value=(0.01, {"weight": "gradient"}, 0.1))
    post_state = SimpleNamespace(params="post_params", opt_state="post_optimizer", step=1)
    optimizer = Mock(return_value=(post_state, 1.0, {"weight": "update"}))
    state, info = agent.update(SimpleNamespace(step=0), "batch", "rng", gradient, optimizer)
    assert state is post_state
    assert info == {"loss": 0.01, "grad_norm": 0.1, "param_norm": 1.0}
    gradient.assert_called_once()
    optimizer.assert_called_once()
    agent.failure.assert_not_called()
