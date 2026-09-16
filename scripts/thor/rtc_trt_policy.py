"""Fixed-shape trained RTC TensorRT adapter; dynamic committed prefix stays data."""

import torch

from onnx_sampler import check_text_bucket
from onnx_sampler import flat_inputs
from rtc_onnx_sampler import RTC_INPUT_NAMES
from rtc_onnx_sampler import validate_trained_prefix
from trt_policy import TensorRTModel

from openpi import transforms
from openpi.policies.policy import Policy


class _NoWeightSampler(torch.nn.Module):
    def sample_actions(self, *args, **kwargs):
        raise RuntimeError("Trained RTC requires the dedicated RTC sampler")


def create_rtc_transform_policy(train_config, norm_stats):
    """Use exactly the Pi/YAM input and inverse transforms without loading weights."""
    data = train_config.data.create(train_config.assets_dirs, train_config.model)
    return Policy(
        _NoWeightSampler(),
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device="cuda",
    )


class RtcTensorRTAdapter:
    def __init__(self, engine_dir, *, max_delay):
        self.model = TensorRTModel(engine_dir, input_names=RTC_INPUT_NAMES)
        if not 0 < max_delay < 50:
            raise ValueError("RTC max_delay must be a trained nonzero delay")
        self.max_delay = max_delay

    def enable_cuda_graph(self):
        self.model.enable_cuda_graph()

    @torch.no_grad()
    def __call__(self, device, observation, *, noise, previous_actions, prefix_mask, num_steps=10):
        if num_steps != 10 or noise is None:
            raise ValueError("RTC TensorRT engine requires ten steps and explicit noise")
        validate_trained_prefix(previous_actions, prefix_mask, max_delay=self.max_delay)
        if self.model.text_bucket < 200:
            check_text_bucket(observation.tokenized_prompt, observation.tokenized_prompt_mask, self.model.text_bucket)
        values = dict(zip(RTC_INPUT_NAMES, (*flat_inputs(observation, noise), previous_actions, prefix_mask), strict=True))
        return self.model.run_flat_inputs(values)
