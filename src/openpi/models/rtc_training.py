"""Training-time action conditioning (arXiv:2512.05964), using OpenPI's clean t=0.

Reference: XenseRobotics-AI/xense-openpi a4e7a304, models/pi0.py.
Unlike that implementation, retain OpenPI's dimension mean and normalize each
example by its postfix length before the trainer takes the batch/time mean.
No new trainable parameters or changes to the action coordinate system.
"""

import jax.numpy as jnp


def condition_prefix(actions, noise, time, delays):
    """Keep the expert prefix clean; each example has an independently sampled delay."""
    prefix = jnp.arange(actions.shape[-2]) < delays[..., None]
    token_time = jnp.where(prefix, 0.0, time[..., None])
    expanded = token_time[..., None]
    return expanded * noise + (1 - expanded) * actions, token_time, prefix


def postfix_loss(loss, prefix):
    """Return [B,H] loss with mean equal to mean of per-example postfix means.

    The caller has already averaged action dimensions, preserving the original
    OpenPI 32D objective (including padded dimensions). Delays must be < H.
    """
    if prefix is None:
        return loss
    postfix = ~prefix
    count = jnp.sum(postfix, axis=-1, keepdims=True)
    return jnp.where(postfix, loss, 0.0) * (loss.shape[-1] / jnp.maximum(count, 1))
