import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("compare_suites", Path(__file__).with_name("compare_suites.py"))
compare_suites = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_suites)


def test_error_metrics_use_absolute_difference_and_preserve_dimensions():
    reference = np.zeros((2, 3, 14), dtype=np.float32)
    candidate = reference.copy()
    candidate[1, 2, 6] = -2
    result = compare_suites.error_metrics(reference, candidate)
    assert result["max_abs"] == 2
    assert result["worst_index"] == [1, 2, 6]
    assert result["per_dimension_max_abs"][6] == 2
    assert result["mae"] == pytest.approx(2 / 84)
    assert result["rmse"] == pytest.approx(np.sqrt(4 / 84))


def test_nonfinite_results_rejected():
    with pytest.raises(ValueError, match="Finite"):
        compare_suites.error_metrics(np.zeros((1, 14)), np.full((1, 14), np.nan))


def test_telemetry_extracts_instantaneous_fields(tmp_path):
    log = tmp_path / "stats"
    log.write_text("RAM 26000/125748MB GR3D_FREQ @[1574,1574,1574] gpu@43.5C/50C VIN 70000mW/40000mW/90000mW\n")
    result = compare_suites.telemetry_metrics(log)
    assert result["gpu_temperature_max_c"] == 43.5
    assert result["input_power_max_w"] == 70
    assert result["system_ram_peak_mib"] == 26000


@pytest.mark.parametrize("changed", ["suite_sha256", "noise_sha256", "norm_stats_sha256", "seed", "versions"])
def test_comparison_rejects_confounded_inputs(monkeypatch, changed):
    keys = (
        "suite_sha256",
        "norm_stats_sha256",
        "checkpoint_metadata_sha256",
        "noise_sha256",
        "seed",
        "steps",
        "horizon",
        "repeats",
        "config",
        "versions",
        "matmul_precision",
    )
    reference = dict.fromkeys(keys, "same")
    candidate = {**reference, changed: "different"}
    monkeypatch.setattr(compare_suites, "read_run", lambda path: (reference if path == "ref" else candidate, {}))
    with pytest.raises(ValueError, match=f"Confounded comparison: {changed}"):
        compare_suites.compare("ref", "candidate")


def test_cross_backend_reports_runtime_changes_but_rejects_changed_workload(monkeypatch):
    keys = (
        "suite_sha256",
        "norm_stats_sha256",
        "checkpoint_metadata_sha256",
        "noise_sha256",
        "seed",
        "steps",
        "horizon",
        "repeats",
        "config",
    )
    reference = {
        **dict.fromkeys(keys, "same"),
        "versions": {"jax": "reference"},
        "matmul_precision": "highest",
        "measurements": [{"sample": "one", "sample_sha256": "same"}],
        "run_id": "reference",
        "p50_ms": 200,
    }
    candidate = {
        **reference,
        "backend": "pytorch",
        "versions": {"torch": "candidate"},
        "matmul_precision": "TF32 disabled",
    }
    arrays = {"actions": np.zeros((1, 2, 50, 14)), "normalized_actions": np.zeros((1, 2, 50, 32))}
    monkeypatch.setattr(compare_suites, "read_run", lambda path: (reference if path == "ref" else candidate, arrays))
    result = compare_suites.compare("ref", "candidate", cross_backend=True)
    assert set(result["runtime_differences"]) == {"versions", "matmul_precision", "backend"}
    assert result["physical_dataset_units"]["max_abs"] == 0
    candidate["horizon"] = 10
    with pytest.raises(ValueError, match="Confounded comparison: horizon"):
        compare_suites.compare("ref", "candidate", cross_backend=True)
