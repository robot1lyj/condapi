import json

import numpy as np
from padding_evidence import validate_padding_evidence
import pytest


def test_padding_evidence_keeps_original_strict_failure_visible(tmp_path):
    expected = {
        "suite_sha256": "suite",
        "norm_stats_sha256": "norm",
        "converted_weights_sha256": "weights",
        "versions": {},
    }
    paths = [tmp_path / name for name in ("control", "fp32", "bf16")]
    for index, (path, dtype, bucket) in enumerate(
        zip(paths, ("float32", "float32", "bfloat16"), (200, 80, 80), strict=True)
    ):
        path.mkdir()
        report = {
            **expected,
            "status": "preparation_diagnosed_not_approved",
            "prepare_only": True,
            "cache_time_modulation": True,
            "compute_dtype": dtype,
            "text_bucket": bucket,
            "tf32": False,
            "quantization": None,
            "noise_sha256": "noise",
            "wrapper_comparisons": [{"sample": "sample.npz", "fp32_diagnostic_close": index == 0}],
        }
        (path / "export_report.json").write_text(json.dumps(report))
        reference = np.zeros((50, 32), dtype=np.float32)
        prepared = reference + (0, 2e-6, 0.002)[index]
        np.savez(path / "sample.npz", reference=reference, prepared=prepared)
    result = validate_padding_evidence(paths, expected, ["sample.npz"], 80)
    assert result["original_strict_pass_count"] == 0
    assert result["comparisons"][0]["framework_fp32_close"]
    assert "not_accuracy_approved" in result["status"]
    with pytest.raises(ValueError, match="Confounded"):
        validate_padding_evidence(paths, {**expected, "suite_sha256": "other"}, ["sample.npz"], 80)
    reference = np.zeros((50, 32), dtype=np.float32)
    np.savez(paths[1] / "sample.npz", reference=reference, prepared=reference + 0.1)
    with pytest.raises(AssertionError):
        validate_padding_evidence(paths, expected, ["sample.npz"], 80)
