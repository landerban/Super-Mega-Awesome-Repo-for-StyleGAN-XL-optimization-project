"""Quantization observers — track running statistics of weights / activations.

Two flavors:
- PerChannelMinMaxObserver: per-output-channel min/max for weight tensors.
- PerTensorEMAMinMaxObserver: per-tensor EMA min/max for activations.
"""

import torch
from torch import nn


class PerChannelMinMaxObserver(nn.Module):
    """Per-output-channel symmetric weight observer.

    Records running min/max along `ch_axis` (default 0 — output channels)
    and exposes `scale_zp(num_bits, symmetric)` to compute INT scales.
    """

    def __init__(self, num_channels: int, ch_axis: int = 0):
        super().__init__()
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.full((num_channels,), float("inf")))
        self.register_buffer("max_val", torch.full((num_channels,), float("-inf")))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.transpose(0, self.ch_axis).reshape(x.shape[self.ch_axis], -1)
        cur_min = x_flat.min(dim=1).values
        cur_max = x_flat.max(dim=1).values
        if not bool(self.initialized):
            self.min_val.copy_(cur_min)
            self.max_val.copy_(cur_max)
            self.initialized.fill_(True)
        else:
            self.min_val.copy_(torch.minimum(self.min_val, cur_min))
            self.max_val.copy_(torch.maximum(self.max_val, cur_max))
        return x

    def scale_zp(self, num_bits: int = 8, symmetric: bool = True):
        if symmetric:
            qmax = 2 ** (num_bits - 1) - 1
            amax = torch.maximum(self.min_val.abs(), self.max_val.abs())
            scale = (amax / qmax).clamp_min(1e-8)
            zp = torch.zeros_like(scale, dtype=torch.int32)
        else:
            qmin, qmax = 0, 2 ** num_bits - 1
            scale = ((self.max_val - self.min_val) / (qmax - qmin)).clamp_min(1e-8)
            zp = torch.round(qmin - self.min_val / scale).to(torch.int32)
        return scale, zp


class PerTensorEMAMinMaxObserver(nn.Module):
    """Per-tensor activation observer with EMA min/max."""

    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur_min = x.detach().min()
        cur_max = x.detach().max()
        if not bool(self.initialized):
            self.min_val.copy_(cur_min)
            self.max_val.copy_(cur_max)
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.min_val.mul_(m).add_(cur_min, alpha=1 - m)
            self.max_val.mul_(m).add_(cur_max, alpha=1 - m)
        return x

    def scale_zp(self, num_bits: int = 8, symmetric: bool = False):
        if symmetric:
            qmax = 2 ** (num_bits - 1) - 1
            amax = torch.max(self.min_val.abs(), self.max_val.abs())
            scale = (amax / qmax).clamp_min(1e-8)
            zp = torch.zeros((), dtype=torch.int32, device=scale.device)
        else:
            qmin, qmax = 0, 2 ** num_bits - 1
            scale = ((self.max_val - self.min_val) / (qmax - qmin)).clamp_min(1e-8)
            zp = torch.round(qmin - self.min_val / scale).to(torch.int32)
        return scale, zp
