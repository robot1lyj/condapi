import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_piper_example() -> dict:
    """Creates a random input example for the Piper policy."""
    return {
        "observation.state": np.random.rand(14),
        "observation.images.top_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "put the pen into the box",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _swap_left_right(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] < 14:
        return values
    right = values[..., :7]
    left = values[..., 7:14]
    rest = values[..., 14:]
    return np.concatenate([left, right, rest], axis=-1)


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType
    base_image_key: str = "observation.images.top_rgb"
    left_wrist_image_key: str = "observation.images.left_wrist"
    right_wrist_image_key: str = "observation.images.right_wrist"
    state_key: str = "observation.state"
    action_key: str = "action"
    prompt_key: str = "prompt"
    swap_left_right: bool = False

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data[self.state_key])
        if self.swap_left_right:
            state = _swap_left_right(state)

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
            if self.swap_left_right:
                actions = _swap_left_right(actions)
            inputs["actions"] = actions

        if self.prompt_key in data:
            inputs["prompt"] = data[self.prompt_key]

        for key in ("episode_index", "frame_index", "episode_length", "stage_progress_gt", "progress"):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    action_dim: int = 14
    swap_left_right: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.action_dim])
        if self.swap_left_right:
            actions = _swap_left_right(actions)
        return {"actions": actions}
