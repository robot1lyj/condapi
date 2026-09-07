"""Flat Pi0.5 sampler retaining BF16/FP32 operations and fixed FP64-derived time embeddings.

No quantization, nan_to_num, FP16 conversion, image removal or horizon change.
The legacy model remains available. Validate this wrapper against it before export.
"""

import torch
from torch import nn

from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
INPUT_NAMES = ("images", "img_masks", "lang_tokens", "lang_masks", "state", "noise")


def fixed_time_schedule(num_steps, *, width, device):
    if num_steps != 10:
        raise ValueError("This export contract fixes exactly ten denoising steps")
    dt = torch.full((), -1.0 / num_steps, dtype=torch.float32, device=device)
    time = torch.ones((), dtype=torch.float32, device=device)
    times, embeddings = [], []
    for _ in range(num_steps):
        times.append(time.clone())
        embeddings.append(
            create_sinusoidal_pos_embedding(
                time.expand(1), width, min_period=4e-3, max_period=4.0, device=device
            ).float()[0]
        )
        time += dt
    return dt, torch.stack(times), torch.stack(embeddings)


def flat_inputs(observation, noise):
    return (
        torch.cat([observation.images[key] for key in IMAGE_KEYS], dim=1),
        torch.stack([observation.image_masks[key] for key in IMAGE_KEYS], dim=1),
        observation.tokenized_prompt,
        observation.tokenized_prompt_mask,
        observation.state,
        noise,
    )


class Pi05OnnxSampler(nn.Module):
    def __init__(self, model):
        super().__init__()
        if not model.pi05 or model.config.action_horizon != 50 or model.config.action_dim != 32:
            raise ValueError("Exporter is scoped to Pi0.5 H50 / 32D")
        self.model = model
        dt, times, embeddings = fixed_time_schedule(
            10, width=model.action_in_proj.out_features, device=next(model.parameters()).device
        )
        self.register_buffer("euler_dt", dt)
        self.register_buffer("times", times)
        self.register_buffer("time_embeddings", embeddings)
        # Direct-on-device constants are also friendly to legacy ONNX tracing.
        model.static_denoising_loop = True

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        views = [images[:, index * 3 : (index + 1) * 3] for index in range(3)]
        masks = [img_masks[:, index] for index in range(3)]
        prefix, pad, att = self.model.embed_prefix(views, masks, lang_tokens, lang_masks)
        positions = torch.cumsum(pad, dim=1) - 1
        attention = self.model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))  # noqa: SLF001
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, cache = self.model.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        actions = noise
        for step in range(10):
            velocity = self.model.denoise_step(
                state,
                pad,
                cache,
                actions,
                self.times[step].expand(images.shape[0]),
                time_embedding=self.time_embeddings[step].expand(images.shape[0], -1),
            )
            actions = actions + self.euler_dt * velocity
        return actions


class FlatSamplerAdapter:
    def __init__(self, sampler):
        self.sampler = sampler

    @torch.no_grad()
    def __call__(self, device, observation, *, noise=None, num_steps=10):
        if num_steps != 10 or noise is None:
            raise ValueError("Export replay requires ten steps and explicit noise")
        inputs = flat_inputs(observation, noise)
        if tuple(inputs[0].shape) != (1, 9, 224, 224) or tuple(noise.shape) != (1, 50, 32):
            raise ValueError("Export replay requires batch 1, three 224 RGB views and H50/32D")
        self.last_inputs = inputs
        return self.sampler(*inputs)
