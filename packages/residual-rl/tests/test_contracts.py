"""Pure-array/data tests only: no Torch import, forward pass, or optimizer."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/residual-rl/src"))
from yam_residual_rl.contracts import bounded_target  # noqa: E402
from yam_residual_rl.contracts import discount_seconds  # noqa: E402
from yam_residual_rl.contracts import progress_reward  # noqa: E402
from yam_residual_rl.contracts import sample_episode_balanced  # noqa: E402


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/residual_rl/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_counts_preserve_partial_success():
    assert progress_reward(13, 9, 2, 2) == 9 / 13
    assert progress_reward(13, 0, 0, 13) == 0
    assert progress_reward(13, 13, 0, 0) == 1
    with pytest.raises(ValueError, match=r"correct.*wrong.*unplaced"):
        progress_reward(13, 13, 1, 0)
    with pytest.raises(ValueError, match="integers"):
        progress_reward(13, newly_correct=True, wrong=0, unplaced=12)


def test_same_real_duration_has_same_discount_at_different_fps():
    assert np.isclose(discount_seconds(1 / 30, 0.99) ** 30, discount_seconds(1 / 10, 0.99) ** 10)


def test_committed_rtc_prefix_is_exact_even_when_residual_is_large():
    base = np.full((5, 14), 0.9)
    lower, upper = np.zeros(14), np.ones(14)
    result = bounded_target(
        base, np.ones_like(base), np.full(14, 0.2), lower, upper, committed=[True, True, False, False, False]
    )
    np.testing.assert_array_equal(result[:2], base[:2])
    np.testing.assert_array_equal(result[2:], np.ones((3, 14)))


def test_invalid_observation_and_epoch_changes_are_not_spliced():
    valid = np.array([True, False, True, True, True, True])
    idx = script("audit_hil").transitions(valid, np.arange(6), np.array([0, 0, 0, 1, 1, 1]), np.arange(6) / 30)
    np.testing.assert_array_equal(idx, [3, 4])


def test_labels_separate_human_help_and_interruption():
    row = {
        "episode_id": "one",
        "initial_remaining": 13,
        "newly_correct": 9,
        "wrong": 1,
        "unplaced": 3,
        "assisted": False,
        "end_reason": "task_budget",
        "base_fingerprint": "fixed",
    }
    labels = script("label_rollouts").validate(
        [row, {**row, "episode_id": "two", "end_reason": "interrupted", "assisted": True}]
    )
    assert labels[0]["reward"] == 9 / 13
    assert labels[1]["reward"] is None
    assert not labels[1]["autonomous"]
    with pytest.raises(ValueError, match="pin one base"):
        script("label_rollouts").validate([row, {**row, "episode_id": "two", "base_fingerprint": "other"}])


def test_slow_hil_does_not_receive_more_episode_sampling_weight():
    episodes, rows = sample_episode_balanced([10, 10000], 10000, np.random.default_rng(12))
    assert 0.48 < np.mean(episodes == 0) < 0.52
    assert np.all(rows[episodes == 0] < 10)


def test_minimal_final_counts_and_pending_review():
    rows = [
        {"episode_id": "one", "newly_correct": 9, "wrong": 2},
        {"episode_id": "two", "end_reason": "pending_review"},
    ]
    result = script("label_rollouts").validate(rows, base_fingerprint="fixed", initial_remaining=13)
    assert result[0]["unplaced"] == 2
    assert result[0]["reward"] == 9 / 13
    assert result[1]["reward"] is None


def test_contract_import_does_not_load_torch():
    # Training remains entirely outside these tests.
    assert "torch" not in sys.modules
