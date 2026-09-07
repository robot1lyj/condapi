import pytest
from thor_triton_autotune import enable_thor_triton_autotune
import torch


@pytest.mark.parametrize(
    ("available", "count", "capability", "name"),
    [
        (False, 0, (11, 0), "NVIDIA Thor"),
        (True, 2, (11, 0), "NVIDIA Thor"),
        (True, 1, (9, 0), "NVIDIA Thor"),
        (True, 1, (11, 0), "Different GPU"),
    ],
)
def test_experiment_rejects_non_thor_or_multi_gpu(monkeypatch, available, count, capability, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: name)
    with pytest.raises(RuntimeError, match="Thor"):
        enable_thor_triton_autotune()
