"""Small CPU tests of the exact converter helpers, without loading a 3B model."""

import ast
from pathlib import Path

import pytest
import torch


def converter_function(name):
    path = Path(__file__).resolve().parents[2] / "examples/convert_jax_model_to_pytorch.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_fp32_loading_preserves_values_that_bf16_would_round():
    model = torch.nn.Linear(2, 1)
    params = {"weight": torch.tensor([[1.0001, -0.1234567]]), "bias": torch.tensor([0.9876543])}
    assert not torch.equal(params["weight"], params["weight"].bfloat16().float())
    result = converter_function("load_mapped_parameters")(model, params)
    assert result["mapped_to_loaded_bit_exact"]
    assert result["mapped_tensor_count"] == 2
    assert torch.equal(model.weight, params["weight"])


@pytest.mark.parametrize("problem", ["missing", "extra", "lora", "destination_bf16", "source_bf16", "nan", "shape"])
def test_converter_rejects_incomplete_or_rounded_parameters(problem):
    model = torch.nn.Linear(2, 1)
    params = {key: value.clone() for key, value in model.state_dict().items()}
    if problem == "missing":
        params.pop("bias")
    elif problem in ("extra", "lora"):
        params[problem] = torch.zeros(1)
    elif problem == "destination_bf16":
        model.bfloat16()
    elif problem == "source_bf16":
        params["weight"] = params["weight"].bfloat16()
    elif problem == "nan":
        params["weight"][0, 0] = float("nan")
    elif problem == "shape":
        params["weight"] = torch.zeros(2, 1)
    with pytest.raises(ValueError, match=r"Incomplete mapping|FP32 source|Invalid source"):
        converter_function("load_mapped_parameters")(model, params)
