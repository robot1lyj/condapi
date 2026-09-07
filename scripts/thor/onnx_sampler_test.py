from types import SimpleNamespace

from onnx_sampler import IMAGE_KEYS
from onnx_sampler import fixed_time_schedule
from onnx_sampler import flat_inputs
import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding


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
