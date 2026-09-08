import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from scripts.thor import compare_suites
from scripts.thor import probe_rlinf


def test_gelu_ablation_is_explicit_and_does_not_patch_global_torch():
    original = functional.gelu
    gemma = SimpleNamespace()
    siglip = SimpleNamespace()
    probe_rlinf.use_tanh_gelu(gemma, siglip)
    values = torch.tensor([-2.0, -0.7, 0.3, 1.8])
    expected = functional.gelu(values, approximate="tanh")
    assert torch.equal(siglip.F.gelu(values), expected)
    assert torch.equal(gemma.gelu_glu(values, torch.ones_like(values)), expected)
    assert functional.gelu is original
    assert siglip.F.conv2d is functional.conv2d


def test_output_records_cannot_overwrite_previous_evidence(tmp_path):
    path = tmp_path / "result.json"
    probe_rlinf.write_json(path, {"value": "first"})
    with pytest.raises(FileExistsError):
        probe_rlinf.write_json(path, {"value": "second"})
    assert "first" in path.read_text()


def test_core_package_import_does_not_execute_parent_package(tmp_path):
    parent = tmp_path / "unwanted_orchestrator"
    package = parent / "model_core"
    package.mkdir(parents=True)
    (parent / "__init__.py").write_text('raise RuntimeError("do not import training stack")')
    (package / "__init__.py").write_text("from .values import value")
    (package / "values.py").write_text("value = 42")
    name = "isolated_probe_test"
    try:
        module = probe_rlinf.load_package(name, package)
        assert module.value == 42
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(f"{name}.values", None)


@pytest.fixture
def paired_reference(monkeypatch):
    reference = {
        **dict.fromkeys(("suite_sha256", "norm_stats_sha256", "checkpoint_metadata_sha256", "noise_sha256"), "same"),
        "steps": 10,
        "horizon": 50,
        "seed": 0,
        "run_id": "jax-reference",
        "measurements": [{"sample_sha256": "one"}, {"sample_sha256": "two"}],
    }
    arrays = {"actions": np.zeros((2, 20, 50, 14)), "normalized_actions": np.zeros((2, 20, 50, 32))}
    monkeypatch.setattr(compare_suites, "read_run", lambda _: (reference, arrays))
    monkeypatch.setattr(probe_rlinf, "digest", lambda _: "fingerprint")
    return reference


def test_precision_comparison_does_not_inflate_sample_count_with_repeats(paired_reference, tmp_path):
    actions = np.zeros((2, 3, 50, 14))
    normalized = np.zeros((2, 3, 50, 32))
    actions[:, 1:] = 100
    result = probe_rlinf.compare_reference(tmp_path, paired_reference, actions, normalized)
    assert result["physical_dataset_units"]["max_abs"] == 0
    assert result["repeat_difference"] == 100


@pytest.mark.parametrize(
    "key",
    [
        "suite_sha256",
        "norm_stats_sha256",
        "checkpoint_metadata_sha256",
        "noise_sha256",
        "steps",
        "horizon",
        "seed",
        "measurements",
    ],
)
def test_probe_rejects_mismatched_reference(paired_reference, key, tmp_path):
    candidate = copy.deepcopy(paired_reference)
    candidate[key] = [{"sample_sha256": "wrong"}] if key == "measurements" else "different"
    with pytest.raises(ValueError, match=r"Confounded reference|sample order mismatch"):
        probe_rlinf.compare_reference(tmp_path, candidate, np.zeros((2, 3, 50, 14)), np.zeros((2, 3, 50, 32)))
