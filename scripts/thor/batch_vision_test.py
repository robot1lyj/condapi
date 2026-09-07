"""Verify view/batch/mask ordering on the real embed_prefix method using tiny CPU embeddings."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("batch_size", [1, 2])
def test_batched_vision_preserves_every_view_and_mask(batch_size):
    source = Path(__file__).resolve().parents[2] / "src/openpi/models_pytorch/pi0_pytorch.py"
    tree = ast.parse(source.read_text())
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PI0Pytorch")
    method = next(
        node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "embed_prefix"
    )
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    calls = []

    def embed_image(image):
        calls.append(image.shape[0])
        return image.mean(dim=(2, 3)).unsqueeze(1).expand(-1, 4, -1)

    model = SimpleNamespace(
        batch_vision=False,
        paligemma_with_expert=SimpleNamespace(
            embed_image=embed_image,
            embed_language_tokens=lambda tokens: tokens.float().unsqueeze(-1).expand(-1, -1, 3),
        ),
        _apply_checkpoint=lambda fn, *args: fn(*args),
    )
    images = [torch.arange(batch_size * 3 * 4 * 4).reshape(batch_size, 3, 4, 4).float() + i * 100 for i in range(3)]
    masks = [torch.full((batch_size,), i != 1, dtype=torch.bool) for i in range(3)]
    tokens = torch.ones(batch_size, 2, dtype=torch.int64)
    token_masks = torch.ones(batch_size, 2, dtype=torch.bool)
    old = namespace["embed_prefix"](model, images, masks, tokens, token_masks)
    assert calls == [batch_size] * 3
    calls.clear()
    model.batch_vision = True
    new = namespace["embed_prefix"](model, images, masks, tokens, token_masks)
    assert calls == [batch_size * 3]
    assert all(torch.equal(a, b) for a, b in zip(old, new, strict=True))
    assert new[0].shape == (batch_size, 14, 3)


def test_native_bf16_mask_preserves_masking_and_softmax_of_bf16_scores():
    source = Path(__file__).resolve().parents[2] / "src/openpi/models_pytorch/pi0_pytorch.py"
    tree = ast.parse(source.read_text())
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PI0Pytorch")
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_attention_masks_4d"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    mask = torch.tensor([[[True, False, True], [False, False, False], [True, True, True]]])
    old = namespace[method.name](SimpleNamespace(attention_mask_dtype=None), mask)
    new = namespace[method.name](SimpleNamespace(attention_mask_dtype=torch.bfloat16), mask)
    assert old.dtype == torch.float32
    assert new.dtype == torch.bfloat16
    assert torch.equal(old == 0, new == 0)
    assert torch.isfinite(new).all()
    scores = torch.linspace(-10, 10, 9).reshape(1, 1, 3, 3).bfloat16()
    assert torch.equal((scores + old).softmax(-1, dtype=torch.float32), (scores + new).softmax(-1, dtype=torch.float32))
