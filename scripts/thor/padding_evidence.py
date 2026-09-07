"""Paired evidence for an offline padding experiment; never task-level approval."""

import json
from pathlib import Path

from benchmark_pi05 import digest
import numpy as np
import torch
from torch.testing._comparison import default_tolerances


def validate_padding_evidence(paths, expected, sample_names, bucket):
    if len(paths) != 3 or bucket >= 200:
        raise ValueError("Need FP32 control, FP32 padding, BF16 padding diagnostic directories")
    reports, fingerprints = [], []
    for source, dtype, expected_bucket in zip(
        paths, ("float32", "float32", "bfloat16"), (200, bucket, bucket), strict=True
    ):
        path = Path(source)
        report_path = path / "export_report.json"
        report = json.loads(report_path.read_text())
        if (
            report["status"] != "preparation_diagnosed_not_approved"
            or not report["prepare_only"]
            or report["compute_dtype"] != dtype
            or report["text_bucket"] != expected_bucket
            or not report["cache_time_modulation"]
            or report["tf32"]
            or report["quantization"] is not None
        ):
            raise ValueError("Wrong padding diagnostic scope")
        for key in ("suite_sha256", "norm_stats_sha256", "converted_weights_sha256", "versions"):
            if report[key] != expected[key]:
                raise ValueError(f"Confounded padding diagnostic: {key}")
        if [row["sample"] for row in report["wrapper_comparisons"]] != sample_names:
            raise ValueError("Diagnostic sample ordering mismatch")
        reports.append(report)
        fingerprints.append({"path": str(path), "report_sha256": digest(report_path), "arrays_sha256": {}})
    if len({report["noise_sha256"] for report in reports}) != 1:
        raise ValueError("Padding diagnostics used different noise")
    framework_rtol, framework_atol = default_tolerances(torch.float32)
    comparisons = []
    for name in sample_names:
        arrays = []
        for path, fingerprint in zip(paths, fingerprints, strict=True):
            array_path = Path(path) / name
            with np.load(array_path, allow_pickle=False) as data:
                reference, prepared = data["reference"], data["prepared"]
            if any(
                x.shape != (50, 32) or x.dtype != np.float32 or not np.isfinite(x).all() for x in (reference, prepared)
            ):
                raise ValueError("Expected finite H50 FP32 diagnostic arrays")
            fingerprint["arrays_sha256"][name] = digest(array_path)
            arrays.append((reference, prepared))
        control, fp32, bf16 = arrays
        if not np.array_equal(control[0], control[1]) or not np.array_equal(control[0], fp32[0]):
            raise ValueError("FP32 control changed or untrimmed references differ")
        # Keep the original stricter diagnostic outcome in its immutable report.
        # This additional, pre-existing framework default is only a numerical
        # check for proceeding with an offline experiment, not a task tolerance.
        torch.testing.assert_close(torch.from_numpy(fp32[1]), torch.from_numpy(fp32[0]))
        comparisons.append(
            {
                "sample": name,
                "fp32_control_exact": True,
                "framework_fp32_close": True,
                "fp32_raw_max_abs": float(np.abs(fp32[1] - fp32[0]).max()),
                "bf16_raw_max_abs": float(np.abs(bf16[1] - bf16[0]).max()),
            }
        )
    return {
        "status": "offline_experiment_supported_not_accuracy_approved",
        "diagnostics": fingerprints,
        "text_bucket": bucket,
        "noise_sha256": reports[0]["noise_sha256"],
        "framework_fp32_tolerance": {"rtol": framework_rtol, "atol": framework_atol},
        "original_strict_pass_count": sum(row["fp32_diagnostic_close"] for row in reports[1]["wrapper_comparisons"]),
        "comparisons": comparisons,
        "scope": "mask/position mapping tested separately; full-text and JAX differences remain explicit",
    }
