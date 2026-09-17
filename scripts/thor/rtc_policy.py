"""Strict trained-prefix RTC inference on Thor; never invoke guidance RTC.

The controller supplies the already committed physical targets aligned to the
first target tick of the new chunk. Thor converts only those targets to the
checkpoint's normalized action space and conditions the denoising postfix.
"""

import time

import numpy as np
import torch

from rtc_action_space import encode_committed_actions
from rtc_onnx_sampler import Pi05RtcOnnxSampler
from rtc_onnx_sampler import RtcFlatSamplerAdapter
from rtc_onnx_sampler import validate_trained_prefix

from openpi.models import model as model_api


def validate_request(observation, rtc, *, max_delay):
    if not isinstance(observation, dict) or not isinstance(rtc, dict):
        raise ValueError("RTC observation and payload must be objects")
    state = np.asarray(observation.get("observation.state"), dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("RTC observation.state must be finite 14D")
    for view in ("top_rgb", "left_rgb", "right_rgb"):
        image = np.asarray(observation.get(f"observation.images.{view}"))
        if image.dtype != np.uint8 or image.ndim != 3 or (image.shape[-1] != 3 and image.shape[0] != 3):
            raise ValueError(f"RTC {view} must be uint8 RGB")
    if not isinstance(observation.get("prompt"), str) or not observation["prompt"].strip():
        raise ValueError("RTC prompt is required")
    delay = rtc.get("delay_steps")
    if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay <= max_delay or delay >= 50:
        raise ValueError("RTC delay_steps exceeds trained range")
    observed_tick = rtc.get("observation_policy_tick")
    target_tick, committed_tick = rtc.get("target_start_tick"), rtc.get("committed_start_tick")
    if isinstance(observed_tick, bool) or not isinstance(observed_tick, int) or observed_tick < 0:
        raise ValueError("RTC observation_policy_tick must be a nonnegative 30Hz policy tick")
    if isinstance(target_tick, bool) or not isinstance(target_tick, int) or target_tick < 0:
        raise ValueError("RTC target_start_tick must be a nonnegative 30Hz policy tick")
    if target_tick != observed_tick:
        raise ValueError("RTC action[0] must target the observation's policy tick")
    if committed_tick != target_tick:
        raise ValueError("RTC committed_start_tick must equal target_start_tick")
    absolute_prefix = np.asarray(rtc.get("committed_actions"), dtype=np.float32)
    if delay == 0 and absolute_prefix.size == 0:
        absolute_prefix = absolute_prefix.reshape(0, 14)
    if absolute_prefix.shape != (delay, 14) or not np.isfinite(absolute_prefix).all():
        raise ValueError("RTC committed_actions must be finite [delay_steps,14] absolute targets")
    return state, absolute_prefix, delay


class RtcEagerAdapter:
    def __init__(self, model, *, max_delay):
        self.model = model
        self.max_delay = max_delay

    @torch.no_grad()
    def __call__(self, device, observation, *, noise, previous_actions, prefix_mask, num_steps=10):
        validate_trained_prefix(previous_actions, prefix_mask, max_delay=self.max_delay)
        result = self.model.sample_actions_trained_rtc(
            device, observation, noise=noise, previous_actions=previous_actions,
            prefix_mask=prefix_mask, num_steps=num_steps,
        )
        self.last_raw = result.detach().clone()
        return result


class TrainedRtcInference:
    """Policy transforms + a dedicated RTC sampler; the ordinary path is untouched."""

    def __init__(self, policy, norm_stats, *, max_delay, text_bucket=200, sampler=None):
        if not 0 < max_delay < 50:
            raise ValueError("A trained RTC checkpoint must specify 0 < max_delay < 50")
        self.policy = policy
        self.norm_stats = norm_stats
        self.max_delay = max_delay
        self.device = torch.device(policy._pytorch_device)  # noqa: SLF001
        self.use_quantiles = False  # pi05_yam contract; assert before serving a different config.
        self.noise_rng = np.random.default_rng()
        self.sampler = sampler or RtcFlatSamplerAdapter(
            Pi05RtcOnnxSampler(policy._model, cache_time_modulation=False, text_bucket=text_bucket).eval(),  # noqa: SLF001
            max_delay=max_delay,
        )

    def infer_rtc(self, obs, rtc, *, noise=None):
        state, physical_prefix, delay = validate_request(obs, rtc, max_delay=self.max_delay)
        # The suffix values of this placeholder are ignored by the prefix mask.
        absolute = np.broadcast_to(state, (50, 14)).copy()
        absolute[:delay] = physical_prefix
        model_prefix = encode_committed_actions(
            absolute, state, self.norm_stats, use_quantiles=self.use_quantiles
        )
        transformed = self.policy._input_transform(dict(obs))  # noqa: SLF001
        tensors = {key: torch.as_tensor(np.asarray(value), device=self.device)[None] for key, value in transformed.items()}
        observation = model_api.Observation.from_dict(tensors)
        if noise is None:
            noise = self.noise_rng.standard_normal((1, 50, 32)).astype(np.float32)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (1, 50, 32) or not np.isfinite(noise).all():
            raise ValueError("RTC noise must be finite [1,50,32]")
        noise = torch.as_tensor(noise, device=self.device)
        previous = torch.as_tensor(model_prefix[None], device=self.device)
        mask = torch.arange(50, device=self.device)[None] < delay
        start = time.monotonic()
        with torch.no_grad():
            result = self.sampler(
                self.device, observation, noise=noise, previous_actions=previous, prefix_mask=mask, num_steps=10
            )
        infer_ms = (time.monotonic() - start) * 1000
        outputs = {"state": np.asarray(tensors["state"][0].cpu()), "actions": np.asarray(result[0].cpu())}
        outputs = self.policy._output_transform(outputs)  # noqa: SLF001
        actions = np.asarray(outputs["actions"])
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise RuntimeError("RTC returned invalid physical H50/14D actions")
        # The controller's already committed commands are authoritative, not a
        # floating-point normalize/inverse-normalize round trip.
        actions[:delay] = physical_prefix
        outputs["actions"] = actions
        outputs["policy_timing"] = {"infer_ms": infer_ms}
        return outputs
