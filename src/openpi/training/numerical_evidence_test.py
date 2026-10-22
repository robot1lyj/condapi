"""CPU array/serialization checks only; no framework, model, or training loop."""

import json
import math

import numpy as np

from openpi.training import numerical_evidence as evidence


def record(*, path="weight", device=0, index=None, maximum=3.0, squares=1.0, nan=0, posinf=0):
    return {
        "path": path,
        "device_id": device,
        "index": index or [[0, 4, None]],
        "integers": [4, nan, posinf, 0, 0 if nan or posinf else -1, 0],
        "floats": [maximum, squares, float("nan") if nan else 0.0, maximum],
    }


def test_stable_norm_preserves_large_finite_values_without_float32_square_overflow():
    result = evidence.summarize_records([record(maximum=1e30, squares=4)])
    assert math.isclose(result["stable_finite_values_l2"], 2e30)
    assert result["all_finite"]
    verdict = evidence.gradient_verdict(0.01, float("inf"), result)
    assert not verdict["update_allowed"]
    assert verdict["finite_elements_but_native_norm_nonfinite"]
    assert "nonfinite_gradient_elements" not in verdict["reasons"]


def test_replicas_do_not_inflate_norm_or_hide_a_bad_replica():
    good = record()
    bad_replica = record(device=1, nan=1)
    result = evidence.summarize_records([good, bad_replica, record(device=2, index=[[4, 8, None]], maximum=4)])
    assert result["stable_finite_values_l2"] == 5
    assert result["logical_shards"] == 2
    assert result["bad_record_indexes"] == [1]
    assert result["replica_disagreement_indexes"] == [1]
    assert not evidence.gradient_verdict(0.01, 5.0, result)["update_allowed"]


def test_nonfinite_loss_is_rejected_even_when_gradients_are_finite():
    result = evidence.summarize_records([record(maximum=0, squares=0)])
    assert evidence.gradient_verdict(0.01, 0.0, result)["update_allowed"]
    assert evidence.gradient_verdict(float("nan"), 0.0, result)["reasons"] == ["nonfinite_loss"]


def test_json_contains_explicit_nonfinite_values_and_can_be_updated_atomically(tmp_path):
    path = tmp_path / "failure.json"
    evidence.write_json(path, {"value": np.array([np.nan, np.inf, -np.inf]), "complete": False})
    assert json.loads(path.read_text())["value"] == ["NaN", "+Inf", "-Inf"]
    evidence.write_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    assert not list(tmp_path.glob("*.writing"))


def test_identity_survives_multiple_transform_wrappers_and_missing_fields_are_explicit():
    identity = evidence.sample_identity({"episode_index": np.array(12), "frame_index": np.array(34)}, 56)
    assert identity["dataset_index"] == 56
    assert identity["episode_index"] == 12
    assert identity["frame_index"] == 34
    assert identity["index"] == -1
    assert math.isnan(identity["timestamp"])
    assert evidence.sample_identity({evidence.IDENTITY_KEY: identity}, 99) is identity


def test_fingerprint_changes_with_values_shape_or_dtype_and_does_not_modify_input():
    array = np.array([1, 2], dtype=np.int32)
    original = array.copy()
    first = evidence.batch_fingerprint({"actions": array})
    assert first == evidence.batch_fingerprint({"actions": array.copy()})
    assert first != evidence.batch_fingerprint({"actions": array.astype(np.float32)})
    assert first != evidence.batch_fingerprint({"actions": array.reshape(1, 2)})
    assert first != evidence.batch_fingerprint({"actions": array + 1})
    np.testing.assert_array_equal(array, original)
