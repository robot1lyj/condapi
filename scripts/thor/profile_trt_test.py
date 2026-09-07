from profile_trt import aggregate_layers
import pytest


def test_aggregate_uses_inference_count_not_callback_count():
    layers = aggregate_layers([("a", 2), ("a", 3), ("b", 1)], 2, {"a": {"LayerType": "MatMul"}})
    assert layers[0]["mean_ms_per_inference"] == 2.5
    assert layers[0]["callback_count"] == 2
    assert layers[0]["layer_type"] == "MatMul"
    assert layers[1]["mean_ms_per_inference"] == 0.5
    assert layers[1]["layer_type"] == "unmatched"
    with pytest.raises(ValueError, match="Nonempty"):
        aggregate_layers([], 0, {})
    with pytest.raises(ValueError, match="Invalid"):
        aggregate_layers([("bad", float("nan"))], 1, {})
