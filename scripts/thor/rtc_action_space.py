"""Map committed YAM absolute targets into the checkpoint's RTC action space.

The previous chunk comes from the controller as absolute 14D actions. RTC must
condition on actions normalized exactly like training: 12 joint deltas from
the *new observation's* state, two absolute grippers, then 32D padding.
"""

import numpy as np

from openpi import transforms
from openpi.policies.yam_policy import YamOutputs

YAM_DELTA_MASK = transforms.make_bool_mask(6, -1, 6, -1)


def encode_committed_actions(absolute_actions, observation_state, norm_stats, *, use_quantiles):
    absolute = np.asarray(absolute_actions, dtype=np.float32)
    state = np.asarray(observation_state, dtype=np.float32)
    if absolute.shape != (50, 14) or state.shape != (14,):
        raise ValueError("RTC requires aligned absolute actions [50,14] and observation state [14]")
    if not np.isfinite(absolute).all() or not np.isfinite(state).all():
        raise ValueError("RTC actions/state must be finite")
    data = {"state": state.copy(), "actions": absolute.copy()}
    data = transforms.DeltaActions(YAM_DELTA_MASK)(data)
    data = transforms.Normalize(norm_stats, use_quantiles=use_quantiles)(data)
    data = transforms.PadStatesAndActions(32)(data)
    encoded = np.asarray(data["actions"], dtype=np.float32)
    if encoded.shape != (50, 32) or not np.isfinite(encoded).all():
        raise ValueError("RTC encoding did not produce finite H50/32D actions")
    return encoded


def decode_model_actions(model_actions, observation_state, norm_stats, *, use_quantiles):
    """Reference inverse for prefix round-trip checks, not a robot safety clamp."""
    actions = np.asarray(model_actions, dtype=np.float32)
    state = np.asarray(observation_state, dtype=np.float32)
    if actions.shape != (50, 32) or state.shape != (14,):
        raise ValueError("RTC inverse requires H50/32D model actions and a 14D state")
    normalized_state = transforms.Normalize(norm_stats, use_quantiles=use_quantiles)({"state": state.copy()})["state"]
    data = {"state": normalized_state, "actions": actions.copy()}
    data = transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)(data)
    data = transforms.AbsoluteActions(YAM_DELTA_MASK)(data)
    return YamOutputs()(data)["actions"]
