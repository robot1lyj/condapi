"""Read recorded inference inputs without importing a model or GPU framework."""

import numpy as np


def read_observation(path):
    with np.load(path, allow_pickle=False) as data:
        observation = {key: np.array(data[key]) for key in data.files}
    state = observation["observation.state"]
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Recorded state must be finite 14D")
    for view in ("top_rgb", "left_rgb", "right_rgb"):
        image = observation[f"observation.images.{view}"]
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Recorded images must be HWC uint8 RGB")
    observation["prompt"] = str(observation["prompt"].item())
    if not observation["prompt"].strip():
        raise ValueError("Recorded observation needs a task prompt")
    return observation
