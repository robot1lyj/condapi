"""Policy transforms for the bimanual YAM LeRobot data contract.

YAM training rows use three RGB streams and one concatenated arm vector:

* ``observation.images.top_rgb``: third-person/top view;
* ``observation.images.left_rgb`` and ``observation.images.right_rgb``: wrist views;
* ``observation.state`` and ``action``: ``[left 6 joints, left gripper, right 6 joints,
  right gripper]``.

The model still uses OpenPI's padded action space internally.  ``YamOutputs`` removes
that padding at the policy boundary and returns the real YAM action dimension.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

YAM_ARM_DIM = 7
YAM_STATE_ACTION_DIM = 2 * YAM_ARM_DIM
YAM_METADATA_KEYS = ("episode_index", "frame_index", "episode_length", "task_index")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"YAM images must be numeric RGB arrays, got dtype={image.dtype}.")
    if not np.isfinite(image).all():
        raise ValueError("YAM images contain NaN or Inf.")
    if np.issubdtype(image.dtype, np.floating):
        if image.size:
            minimum = float(np.nanmin(image))
            maximum = float(np.nanmax(image))
            if minimum >= 0.0 and maximum <= 1.0:
                image = 255 * image
            elif minimum >= -1.0 and maximum <= 1.0 and minimum < 0.0:
                image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"YAM images must be HWC or CHW RGB, got shape={image.shape}.")
    return image


def _validate_yam_dim(name: str, value: np.ndarray, expected_dim: int) -> None:
    if value.ndim == 0 or value.shape[-1] != expected_dim:
        raise ValueError(f"YAM {name} must be {expected_dim}D, got shape={value.shape}.")


def _validate_action_dim(action_dim: int) -> None:
    if action_dim <= 0 or action_dim % YAM_ARM_DIM != 0:
        raise ValueError(f"YAM action_dim must be a positive multiple of {YAM_ARM_DIM}, got {action_dim}.")


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Convert YAM LeRobot rows or inference observations to OpenPI inputs."""

    model_type: _model.ModelType
    action_dim: int = YAM_STATE_ACTION_DIM
    base_image_key: str = "observation.images.top_rgb"
    left_wrist_image_key: str = "observation.images.left_rgb"
    right_wrist_image_key: str = "observation.images.right_rgb"
    state_key: str = "observation.state"
    action_key: str = "action"
    prompt_key: str = "prompt"

    def __post_init__(self) -> None:
        _validate_action_dim(self.action_dim)

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data[self.state_key])
        _validate_yam_dim("state", state, self.action_dim)

        images = {
            "base_0_rgb": _parse_image(data[self.base_image_key]),
            "left_wrist_0_rgb": _parse_image(data[self.left_wrist_image_key]),
            "right_wrist_0_rgb": _parse_image(data[self.right_wrist_image_key]),
        }
        image_mask = dict.fromkeys(images, np.True_)

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_mask,
        }

        if self.action_key in data:
            actions = np.asarray(data[self.action_key])
            _validate_yam_dim("actions", actions, self.action_dim)
            inputs["actions"] = actions

        if self.prompt_key in data:
            inputs["prompt"] = data[self.prompt_key]

        for key in YAM_METADATA_KEYS:
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Strip OpenPI model padding from an action chunk for YAM."""

    action_dim: int = YAM_STATE_ACTION_DIM

    def __post_init__(self) -> None:
        _validate_action_dim(self.action_dim)

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim == 0 or actions.shape[-1] < self.action_dim:
            raise ValueError(f"YAM model actions must have at least {self.action_dim} dims, got shape={actions.shape}.")
        if not np.isfinite(actions).all():
            raise ValueError("YAM model actions contain NaN or Inf.")
        return {"actions": actions[..., : self.action_dim]}
