import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

OPENARM_STATE_ACTION_DIM = 16
OPENARM_METADATA_KEYS = ("episode_index", "frame_index", "episode_length", "stage_progress_gt", "progress")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _validate_openarm_dim(name: str, value: np.ndarray) -> None:
    if value.shape[-1] != OPENARM_STATE_ACTION_DIM:
        raise ValueError(f"OpenArm {name} must be {OPENARM_STATE_ACTION_DIM}D, got shape={value.shape}.")


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """Convert OpenArm LeRobot rows into OpenPI model inputs.

    OpenArm uses the HQ 16D contract:
    [right arm 7 joints deg, right gripper motor deg, left arm 7 joints deg, left gripper motor deg].
    """

    model_type: _model.ModelType
    base_image_key: str = "observation.images.base"
    left_wrist_image_key: str = "observation.images.left_wrist"
    right_wrist_image_key: str = "observation.images.right_wrist"
    state_key: str = "observation.state"
    action_key: str = "action"
    prompt_key: str = "prompt"

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data[self.state_key])
        _validate_openarm_dim("state", state)

        base_image = _parse_image(data[self.base_image_key])

        left_exists = self.left_wrist_image_key in data
        right_exists = self.right_wrist_image_key in data
        left_wrist = _parse_image(data[self.left_wrist_image_key]) if left_exists else np.zeros_like(base_image)
        right_wrist = _parse_image(data[self.right_wrist_image_key]) if right_exists else np.zeros_like(base_image)

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_ if left_exists else np.False_,
            "right_wrist_0_rgb": np.True_ if right_exists else np.False_,
        }
        if self.model_type == _model.ModelType.PI0_FAST:
            image_mask = dict.fromkeys(image_mask, np.True_)

        history_image_keys = {
            "base_-100_rgb": f"his_-100_{self.base_image_key}",
            "left_wrist_-100_rgb": f"his_-100_{self.left_wrist_image_key}",
            "right_wrist_-100_rgb": f"his_-100_{self.right_wrist_image_key}",
        }
        for output_key, input_key in history_image_keys.items():
            if input_key in data:
                images[output_key] = _parse_image(data[input_key])
                image_mask[output_key] = np.True_

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_mask,
        }

        if self.action_key in data:
            actions = np.asarray(data[self.action_key])
            _validate_openarm_dim("actions", actions)
            inputs["actions"] = actions

        if self.prompt_key in data:
            inputs["prompt"] = data[self.prompt_key]

        for key in OPENARM_METADATA_KEYS:
            if key in data:
                inputs[key] = data[key]
        inputs.update({key: value for key, value in data.items() if key.startswith("complementary_info.")})

        return inputs


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """Convert model 32D action chunks back to OpenArm 16D robot commands."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < OPENARM_STATE_ACTION_DIM:
            raise ValueError(
                f"OpenArm model actions must have at least {OPENARM_STATE_ACTION_DIM} dims, got shape={actions.shape}."
            )
        return {"actions": actions[..., :OPENARM_STATE_ACTION_DIM]}
