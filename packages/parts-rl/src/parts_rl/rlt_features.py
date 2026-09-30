"""Scoped final-prefix capture shared by offline extraction and eager serving.

No Torch import: protocol tests use inert array fixtures, not model execution.
The readout callable is supplied by the model environment.
"""

from contextlib import contextmanager

import numpy as np


class FrozenRltFeatures:
    def __init__(self, model, readout=None, *, prefix_sink=None):
        self.model, self.readout, self.prefix_sink = model, readout, prefix_sink
        self.z = None
        if not callable(getattr(model, "embed_prefix", None)) or not hasattr(model, "paligemma_with_expert"):
            raise ValueError("RLT needs an eager Pi final-prefix outlet")

    @contextmanager
    def capture(self):
        self.z = None
        state = {"count": None, "mask": None, "captured": False}
        expert = self.model.paligemma_with_expert
        original_embed, original_forward = self.model.embed_prefix, expert.forward
        embed_owned = "embed_prefix" in self.model.__dict__
        forward_owned = "forward" in expert.__dict__

        def embed(images, masks, tokens, token_masks):
            result = original_embed(images, masks, tokens, token_masks)
            count = result[0].shape[1] - tokens.shape[1]
            if len(images) != 3 or count <= 0 or count % 3 or result[0].shape[0] != 1:
                raise ValueError("RLT requires batch-one YAM three-view prefix")
            state.update(count=count, mask=result[1][:, :count])
            return result

        def forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            inputs = kwargs.get("inputs_embeds")
            if inputs is not None and inputs[0] is not None and inputs[1] is None:
                if state["captured"] or state["count"] is None:
                    raise ValueError("Ambiguous/stale RLT final prefix")
                prefix = result[0][0][:, : state["count"]].detach()
                mask = state["mask"].detach()
                if prefix.shape != (1, state["count"], 2048):
                    raise ValueError("Unexpected Pi final-prefix shape")
                state["captured"] = True
                if self.prefix_sink is not None:
                    self.prefix_sink(prefix, mask)
                if self.readout is not None:
                    self.z = np.asarray(self.readout(prefix, mask), dtype=np.float32)
                    if self.z.ndim != 1 or not np.isfinite(self.z).all():
                        raise ValueError("Invalid RL token readout")
            return result

        self.model.embed_prefix, expert.forward = embed, forward
        try:
            yield self
        finally:
            if embed_owned:
                self.model.embed_prefix = original_embed
            else:
                del self.model.embed_prefix
            if forward_owned:
                expert.forward = original_forward
            else:
                del expert.forward
