"""Trained-prefix RTC sampler with fixed shapes and dynamic prefix values.

This is not inference-time gradient guidance. The committed prefix is clean at
flow time zero on every denoising step; only the postfix follows the FP32 Euler
schedule. A fixed [1,50] mask keeps ONNX/TensorRT/CUDA Graph shapes reusable.
"""

import torch
from torch import nn

from onnx_sampler import INPUT_NAMES
from onnx_sampler import Pi05OnnxSampler
from onnx_sampler import check_text_bucket
from onnx_sampler import flat_inputs
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

RTC_INPUT_NAMES = (*INPUT_NAMES, "previous_actions", "prefix_mask")


def validate_trained_prefix(previous, mask, *, max_delay):
    """Validate the model-space H50/32D prefix; never silently clamp delay."""
    if tuple(previous.shape) != (1, 50, 32) or tuple(mask.shape) != (1, 50) or mask.dtype != torch.bool:
        raise ValueError("RTC requires previous_actions [1,50,32] and bool prefix_mask [1,50]")
    if not torch.isfinite(previous).all():
        raise ValueError("RTC prefix contains NaN or Inf")
    delay = int(mask.sum().item())
    if not 0 <= delay <= max_delay or not torch.equal(mask[0], torch.arange(50, device=mask.device) < delay):
        raise ValueError("RTC prefix must be contiguous and within the trained delay range")
    return delay


class CachedRtcProjection(nn.Module):
    """Cache both clean t=0 and each postfix timestep for every adaRMS layer."""

    def __init__(self, original, clean_condition, step_conditions):
        super().__init__()
        if any(p.dtype != torch.float32 for p in original.parameters()) or len(step_conditions) not in (5, 6, 7, 8, 10):
            raise ValueError("RTC modulation cache requires FP32 projection and supported steps")
        self.original = original
        self.step = None
        self.mask = None
        with torch.no_grad():
            clean = original(clean_condition)
            table = torch.stack([original(condition) for condition in step_conditions])
        if not torch.isfinite(clean).all() or not torch.isfinite(table).all():
            raise ValueError("Non-finite RTC modulation cache")
        self.register_buffer("clean", clean, persistent=False)
        self.register_buffer("table", table, persistent=False)

    def forward(self, condition):
        if self.step is None:
            return self.original(condition)
        if self.training or tuple(condition.shape[:2]) != (1, 50) or self.mask is None:
            raise ValueError("RTC modulation cache requires batch-1 H50 inference")
        return torch.where(self.mask[..., None], self.clean[:, None, :], self.table[self.step][:, None, :])


class Pi05RtcOnnxSampler(Pi05OnnxSampler):
    """Separate trained-RTC engine candidate; never modifies the ordinary W sampler."""

    def __init__(self, model, *, cache_time_modulation=False, text_bucket=200, num_steps=10):
        super().__init__(model, cache_time_modulation=False, text_bucket=text_bucket, num_steps=num_steps)
        self.clean_time_embedding = create_sinusoidal_pos_embedding(
            torch.zeros(1, dtype=torch.float32, device=self.times.device),
            model.action_in_proj.out_features,
            4e-3,
            4.0,
            device=self.times.device,
        ).float()[0]
        if cache_time_modulation:
            with torch.no_grad():
                def condition(embedding):
                    return torch.nn.functional.silu(
                        model.time_mlp_out(torch.nn.functional.silu(model.time_mlp_in(embedding[None])))
                    )

                clean = condition(self.clean_time_embedding)
                steps = [condition(embedding) for embedding in self.time_embeddings]
                expert = model.paligemma_with_expert.gemma_expert.model
                for name, module in list(expert.named_modules()):
                    dense = getattr(module, "dense", None)
                    if type(module).__name__ == "GemmaRMSNorm" and isinstance(dense, nn.Linear):
                        cached = CachedRtcProjection(dense, clean, steps)
                        module.dense = cached
                        self.cached_modulations.append(cached)
                        self.cache_names.append(name)
                if len(self.cached_modulations) != 2 * len(expert.layers) + 1:
                    raise ValueError("Unexpected RTC adaptive projection count")

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise, previous_actions, prefix_mask):
        lang_tokens = lang_tokens[:, : self.text_bucket]
        lang_masks = lang_masks[:, : self.text_bucket]
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
        fixed_prefix = prefix_mask[..., None]
        actions = torch.where(fixed_prefix, previous_actions, noise)
        try:
            for step in range(self.num_steps):
                for projection in self.cached_modulations:
                    projection.step = step
                    projection.mask = prefix_mask
                token_time = torch.where(prefix_mask, 0.0, self.times[step])
                token_embedding = torch.where(
                    fixed_prefix,
                    self.clean_time_embedding[None, None, :],
                    self.time_embeddings[step][None, None, :],
                )
                velocity = self.model.denoise_step(
                    state, pad, cache, actions, token_time, time_embedding=token_embedding
                )
                actions = torch.where(fixed_prefix, previous_actions, actions + self.euler_dt * velocity)
        finally:
            for projection in self.cached_modulations:
                projection.step = None
                projection.mask = None
        return actions


class RtcFlatSamplerAdapter:
    def __init__(self, sampler, *, max_delay):
        self.sampler = sampler
        self.max_delay = max_delay

    @torch.no_grad()
    def __call__(self, device, observation, *, noise, previous_actions, prefix_mask, num_steps=10):
        if num_steps != self.sampler.num_steps or noise is None:
            raise ValueError("RTC export requires matching denoising steps and explicit noise")
        validate_trained_prefix(previous_actions, prefix_mask, max_delay=self.max_delay)
        inputs = (*flat_inputs(observation, noise), previous_actions, prefix_mask)
        check_text_bucket(inputs[2], inputs[3], self.sampler.text_bucket)
        if tuple(inputs[0].shape) != (1, 9, 224, 224) or tuple(noise.shape) != (1, 50, 32):
            raise ValueError("RTC export requires batch 1, three 224 RGB views and H50/32D")
        self.last_inputs = inputs
        result = self.sampler(*inputs)
        self.last_raw = result.detach().clone()
        return result
