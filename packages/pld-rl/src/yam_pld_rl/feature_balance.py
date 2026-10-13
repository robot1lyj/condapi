"""Fixed train-only block RMS scaling; preserves the full recorded RTC state."""

import math

import torch
from torch import nn


class FeatureBalance(nn.Module):
    def __init__(self, dim, rms):
        super().__init__()
        sizes = (1536, 14, 700, 14, 23)
        if dim != sum(sizes) or len(rms) != len(sizes) or any(not math.isfinite(v) or v <= 0 for v in rms):
            raise ValueError("feature balance requires complete v3 block scales")
        self.register_buffer("gain", torch.cat([torch.full((n,), 1/scale) for n, scale in zip(sizes, rms, strict=True)]))

    def forward(self, x):
        return x * self.gain


class BalancedQ(nn.Module):
    def __init__(self, q, dim, rms):
        super().__init__()
        self.q = q
        self.balance = FeatureBalance(dim, rms)

    def forward(self, x, action):
        return self.q(self.balance(x), action)
