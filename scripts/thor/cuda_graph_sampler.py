"""Opt-in, single-client CUDA graph replay of the complete fixed-step sampler.

No weight conversion, quantization, input caching or skipped denoising. All
observation tensors and explicit noise are copied on EVERY call. CPU policy
transforms stay outside the graph and inside benchmark latency measurements.

API reference: https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
"""

import jax
import torch


def tensor_signature(leaves):
    if not leaves or any(not isinstance(x, torch.Tensor) for x in leaves):
        raise ValueError("Graph inputs must be a nonempty tensor pytree")
    return [(tuple(x.shape), x.dtype, x.device) for x in leaves]


class CudaGraphSampler:
    def __init__(self, sampler, *, reference_sampler=None):
        self.sampler = sampler
        self.reference_sampler = reference_sampler or sampler
        self.graph = None

    @torch.no_grad()
    def __call__(self, device, observation, *, noise=None, num_steps=10):
        if noise is None:
            raise ValueError("Graph benchmark requires explicit fresh noise")
        leaves, structure = jax.tree.flatten((observation, noise))
        signature = tensor_signature(leaves)
        if not all(x.is_cuda for x in leaves):
            raise ValueError("All graph inputs must already be CUDA tensors")
        if self.graph is None:
            self.signature = signature
            self.structure = structure
            self.device = device
            self.num_steps = num_steps
            self.static_leaves = [x.clone() for x in leaves]
            self.static_observation, self.static_noise = jax.tree.unflatten(structure, self.static_leaves)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._eager()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._eager()
        elif (
            signature != self.signature
            or structure != self.structure
            or device != self.device
            or num_steps != self.num_steps
        ):
            raise ValueError("Graph contract changed; create a new graph explicitly")
        for target, source in zip(self.static_leaves, leaves, strict=True):
            target.copy_(source)
        self.graph.replay()
        # Never expose an output buffer overwritten by the next observation.
        return self.static_output.clone()

    def _eager(self):
        return self.sampler(self.device, self.static_observation, noise=self.static_noise, num_steps=self.num_steps)

    @torch.no_grad()
    def validate_current(self):
        """Compare this observation's graph output to eager, outside timed calls."""
        captured = self.static_output.clone()
        reference = self.reference_sampler(
            self.device, self.static_observation, noise=self.static_noise, num_steps=self.num_steps
        )
        if not torch.isfinite(reference).all() or not torch.isfinite(captured).all():
            raise RuntimeError("Non-finite CUDA graph validation output")
        return float((captured.float() - reference.float()).abs().max().item())
