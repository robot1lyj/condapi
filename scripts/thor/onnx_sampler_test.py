import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

from onnx_sampler import IMAGE_KEYS
from onnx_sampler import CachedTimeProjection
from onnx_sampler import TextBucketSampler
from onnx_sampler import check_text_bucket
from onnx_sampler import fixed_time_schedule
from onnx_sampler import flat_inputs
import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


def test_fixed_schedule_keeps_original_fp32_recurrence():
    device = torch.device("cpu")
    dt, times, embeddings = fixed_time_schedule(10, width=8, device=device)
    time = torch.tensor(1.0, dtype=torch.float32)
    old_dt = torch.tensor(-0.1, dtype=torch.float32)
    index = 0
    while time >= -old_dt / 2:
        expected = create_sinusoidal_pos_embedding(time.expand(1), 8, 4e-3, 4.0, device=device).float()[0]
        assert torch.equal(times[index], time)
        assert torch.equal(embeddings[index], expected)
        time += old_dt
        index += 1
    assert index == 10
    assert torch.equal(dt, old_dt)
    with pytest.raises(ValueError, match="ten"):
        fixed_time_schedule(5, width=8, device=device)


def test_flat_inputs_keeps_all_views_masks_state_and_noise():
    observation = SimpleNamespace(
        images={key: torch.full((1, 3, 4, 4), float(index)) for index, key in enumerate(IMAGE_KEYS)},
        image_masks={key: torch.tensor([index != 1]) for index, key in enumerate(IMAGE_KEYS)},
        tokenized_prompt=torch.ones(1, 8, dtype=torch.int64),
        tokenized_prompt_mask=torch.ones(1, 8, dtype=torch.bool),
        state=torch.arange(32).float()[None],
    )
    noise = torch.zeros(1, 50, 32)
    images, masks, tokens, token_masks, state, actual_noise = flat_inputs(observation, noise)
    for index, key in enumerate(IMAGE_KEYS):
        assert torch.equal(images[:, index * 3 : (index + 1) * 3], observation.images[key])
    assert masks.tolist() == [[True, False, True]]
    assert tokens is observation.tokenized_prompt
    assert token_masks is observation.tokenized_prompt_mask
    assert state is observation.state
    assert actual_noise is noise


def test_precomputed_time_override_preserves_real_embed_suffix_method():
    torch.manual_seed(42)
    model = SimpleNamespace(
        pi05=True,
        static_denoising_loop=False,
        config=SimpleNamespace(action_horizon=50),
        action_in_proj=torch.nn.Linear(32, 8),
        time_mlp_in=torch.nn.Linear(8, 8),
        time_mlp_out=torch.nn.Linear(8, 8),
        _apply_checkpoint=lambda fn, *args: fn(*args),
    )
    state, noise, time = torch.zeros(1, 32), torch.randn(1, 50, 32), torch.tensor([0.3])
    expected = PI0Pytorch.embed_suffix(model, state, noise, time)
    embedding = create_sinusoidal_pos_embedding(time, 8, 4e-3, 4.0, device=time.device).float()
    actual = PI0Pytorch.embed_suffix(model, state, noise, time, time_embedding=embedding)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_adaptive_rmsnorm_repr_does_not_require_nonexistent_weight():
    source = (
        Path(__file__).resolve().parents[2]
        / "src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py"
    )
    cls = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "GemmaRMSNorm"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "extra_repr")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    norm = SimpleNamespace(dim=8, eps=1e-6, cond_dim=4, dense=object())
    result = namespace["extra_repr"](norm)
    assert "8" in result
    assert "cond_dim=4" in result


def test_time_projection_cache_preserves_fp32_and_default_path():
    torch.manual_seed(4)
    linear = torch.nn.Linear(8, 24).eval()
    conditions = [torch.randn(1, 8) for _ in range(10)]
    cached = CachedTimeProjection(linear, conditions).eval()
    for step, condition in enumerate(conditions):
        cached.step = step
        assert torch.equal(cached(condition), linear(condition))
    cached.step = None
    different = torch.randn(2, 8)
    assert torch.equal(cached(different), linear(different))
    assert cached.table.dtype == torch.float32
    assert "table" not in cached.state_dict()
    cached.step = 0
    with pytest.raises(ValueError, match="batch-1"):
        cached(different)
    cached.train()
    with pytest.raises(ValueError, match="batch-1"):
        cached(conditions[0])


def test_text_bucket_rejects_real_token_truncation():
    tokens = torch.arange(200)[None]
    mask = torch.zeros(1, 200, dtype=torch.bool)
    mask[:, :70] = True
    check_text_bucket(tokens, mask, 80)
    mask[:, 199] = True  # Count alone is insufficient: a valid tail token cannot be dropped.
    with pytest.raises(ValueError, match="Valid tokens"):
        check_text_bucket(tokens, mask, 80)
    check_text_bucket(tokens, mask, 200)
    with pytest.raises(ValueError, match="outside"):
        check_text_bucket(tokens, mask, 201)
    with pytest.raises(ValueError, match="boolean"):
        check_text_bucket(tokens, mask.float(), 80)


def test_text_bucket_adapter_preserves_original_observation_and_noise():
    @dataclasses.dataclass
    class Observation:
        tokenized_prompt: torch.Tensor
        tokenized_prompt_mask: torch.Tensor
        images: dict
        state: torch.Tensor

    observation = Observation(torch.arange(200)[None], torch.arange(200)[None] < 70, {}, torch.zeros(1, 32))
    noise = torch.zeros(1, 50, 32)

    def sampler(device, trimmed, *, noise, num_steps):
        assert device == "cpu"
        assert num_steps == 10
        assert trimmed.images is observation.images
        assert trimmed.state is observation.state
        assert trimmed.tokenized_prompt.shape == (1, 80)
        return noise

    adapter = TextBucketSampler(sampler, 80)
    assert adapter("cpu", observation, noise=noise, num_steps=10) is noise
    assert observation.tokenized_prompt.shape == (1, 200)
    observation.tokenized_prompt_mask[:, 199] = True
    with pytest.raises(ValueError, match="Valid tokens"):
        adapter("cpu", observation, noise=noise, num_steps=10)


@pytest.mark.parametrize("valid_text", [64, 70, 80])
def test_padding_compaction_preserves_attention_edges_and_positions(valid_text):
    image_pad = torch.ones(1, 768, dtype=torch.bool)
    text_pad = torch.arange(200)[None] < valid_text
    action_pad = torch.ones(1, 50, dtype=torch.bool)
    full_pad = torch.cat([image_pad, text_pad, action_pad], dim=1)
    full_groups = torch.zeros_like(full_pad)
    full_groups[:, 968] = True
    keep = torch.cat([torch.arange(848), torch.arange(968, 1018)])
    compact_pad = full_pad[:, keep]
    compact_groups = full_groups[:, keep]
    full_mask = make_att_2d_masks(full_pad, full_groups)
    compact_mask = make_att_2d_masks(compact_pad, compact_groups)
    assert torch.equal(full_mask[:, keep][:, :, keep], compact_mask)
    assert torch.equal((torch.cumsum(full_pad, dim=-1) - 1)[:, keep], torch.cumsum(compact_pad, dim=-1) - 1)
