from types import SimpleNamespace

import numpy as np
from onnx_sampler import IMAGE_KEYS
import pytest
import torch
import trt_policy

from openpi import transforms


def test_export_reference_comparison(tmp_path):
    path = tmp_path / "sample.npz"
    reference = np.zeros((50, 32), dtype=np.float32)
    np.savez(path, reference=reference, prepared=reference)
    observed = np.zeros((2, 50, 32), dtype=np.float32)
    assert trt_policy.compare_export_reference(path, observed)["exact"]
    observed[:, 0, 31] = 0.5
    result = trt_policy.compare_export_reference(path, observed)
    assert result["normalized_32d_max_abs"] == 0.5
    assert result["normalized_14d_max_abs"] == 0
    assert not result["exact"]
    observed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        trt_policy.compare_export_reference(path, observed)
    np.savez(path, reference=reference, prepared=reference + 1)
    with pytest.raises(ValueError, match="equivalent"):
        trt_policy.compare_export_reference(path, observed)


def test_factory_preserves_yam_transform_order(monkeypatch):
    data = SimpleNamespace(
        data_transforms=transforms.Group(inputs=["data_in"], outputs=["data_out"]),
        model_transforms=transforms.Group(inputs=["model_in"], outputs=["model_out"]),
        use_quantile_norm=True,
    )
    train = SimpleNamespace(
        data=SimpleNamespace(create=lambda *args: data), assets_dirs=None, model=None, policy_metadata={"model": "pi05"}
    )
    monkeypatch.setattr(trt_policy, "TensorRTModel", lambda path: "engine_model")
    monkeypatch.setattr(trt_policy, "Policy", lambda model, **kwargs: {"model": model, **kwargs})
    policy = trt_policy.create_trt_policy(train, "engine", {})
    assert policy["model"] == "engine_model"
    incoming, outgoing = policy["transforms"], policy["output_transforms"]
    assert isinstance(incoming[0], transforms.InjectDefaultPrompt)
    assert incoming[1] == "data_in"
    assert isinstance(incoming[2], transforms.Normalize)
    assert incoming[2].use_quantiles
    assert incoming[3] == "model_in"
    assert outgoing[0] == "model_out"
    assert isinstance(outgoing[1], transforms.Unnormalize)
    assert outgoing[1].use_quantiles
    assert outgoing[2] == "data_out"
    assert policy["sample_kwargs"] == {"num_steps": 10}


def test_runtime_shape_guard_and_output_ownership(monkeypatch):
    model = trt_policy.TensorRTModel.__new__(trt_policy.TensorRTModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")  # Fake runtime only; real constructor requires CUDA.
    model.inputs = {"noise": ((1, 50, 32), torch.float32)}
    model.unused_bindings = {}
    model.graph_inputs = None
    model.cuda_graph = None
    model.outputs = {"actions": torch.ones(1, 50, 32)}
    model.context = SimpleNamespace(set_tensor_address=lambda *args: True, execute_async_v3=lambda **kwargs: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=7))
    observation = SimpleNamespace(
        images={key: torch.zeros(1, 3, 224, 224) for key in IMAGE_KEYS},
        image_masks={key: torch.ones(1, dtype=torch.bool) for key in IMAGE_KEYS},
        tokenized_prompt=torch.ones(1, 200, dtype=torch.int64),
        tokenized_prompt_mask=torch.ones(1, 200, dtype=torch.bool),
        state=torch.zeros(1, 32),
    )
    noise = torch.zeros(1, 50, 32)
    output = model.sample_actions("cpu", observation, noise=noise)
    model.outputs["actions"].zero_()
    assert output.sum() == 50 * 32
    model.inputs["state"] = ((1, 32), torch.float32)
    model.unused_bindings["state"] = torch.zeros(1, 32, dtype=torch.float32)
    observation.state = observation.state.double()
    model.sample_actions("cpu", observation, noise=noise)
    assert observation.state.dtype == torch.float64
    with pytest.raises(ValueError, match="contract mismatch"):
        model.sample_actions("cpu", observation, noise=noise[:, :10])
    with pytest.raises(ValueError, match="ten denoising"):
        model.sample_actions("cpu", observation, noise=noise, num_steps=5)
    model.context.execute_async_v3 = lambda **kwargs: False
    with pytest.raises(RuntimeError, match="execution failed"):
        model.sample_actions("cpu", observation, noise=noise)
