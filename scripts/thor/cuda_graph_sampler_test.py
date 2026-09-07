"""CPU checks for graph contracts and the actual sampler's fixed Euler loop."""

import ast
from pathlib import Path
from types import SimpleNamespace

from cuda_graph_sampler import CudaGraphSampler
from cuda_graph_sampler import tensor_signature
import pytest
import torch


def test_graph_requires_explicit_noise_and_cuda():
    sampler = CudaGraphSampler(lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="explicit fresh noise"):
        sampler("cpu", torch.ones(1))
    with pytest.raises(ValueError, match="CUDA tensors"):
        sampler("cpu", torch.ones(1), noise=torch.ones(1))


def test_signature_distinguishes_shape_and_dtype():
    signature = tensor_signature([torch.ones(1, 50, 32)])
    assert signature != tensor_signature([torch.ones(1, 10, 32)])
    assert signature != tensor_signature([torch.ones(1, 50, 32).bfloat16()])
    with pytest.raises(ValueError, match="tensor pytree"):
        tensor_signature([None])


@pytest.mark.parametrize("num_steps", [1, 5, 10, 20])
def test_static_loop_preserves_fp32_timestamps_and_every_euler_step(num_steps):
    source = Path(__file__).resolve().parents[2] / "src/openpi/models_pytorch/pi0_pytorch.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI0Pytorch")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_actions")
    namespace = {"torch": torch, "Tensor": torch.Tensor, "make_att_2d_masks": lambda pad, att: pad}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    state = torch.zeros(1, 32)
    times = []

    def denoise(_state, _mask, _cache, x, time):
        times.append(time.clone())
        return x.sin() * time[:, None, None] + 0.125

    model = SimpleNamespace(
        static_denoising_loop=False,
        attention_implementation="eager",
        _preprocess_observation=lambda *args, **kwargs: ([], [], None, None, state),
        embed_prefix=lambda *args: (torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), None),
        _prepare_attention_masks_4d=lambda mask: mask,
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
            forward=lambda **kwargs: (None, None),
        ),
        denoise_step=denoise,
    )
    noise = torch.linspace(-3, 3, 50 * 32).reshape(1, 50, 32)
    old = namespace[method.name](model, "cpu", SimpleNamespace(state=state), noise=noise, num_steps=num_steps)
    old_times = times.copy()
    times.clear()
    model.static_denoising_loop = True
    new = namespace[method.name](model, "cpu", SimpleNamespace(state=state), noise=noise, num_steps=num_steps)
    assert len(times) == len(old_times) == num_steps
    assert all(torch.equal(a, b) for a, b in zip(old_times, times, strict=True))
    assert torch.equal(old, new)
    assert torch.equal(noise, torch.linspace(-3, 3, 50 * 32).reshape(1, 50, 32))
