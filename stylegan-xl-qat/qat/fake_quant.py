"""Fake quantization with the straight-through estimator (STE).

`fake_quantize` rounds + clips in the forward pass while passing the gradient
through unchanged on the backward pass. This is the standard QAT primitive.
"""

import torch
from torch import nn
from torch.autograd import Function


class _FakeQuantSTE(Function):
    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax):
        q = torch.round(x / scale + zp).clamp(qmin, qmax)
        return (q - zp) * scale

    @staticmethod
    def backward(ctx, grad_out):
        # Straight-through: pass gradient on x; no grad for scale/zp/qmin/qmax.
        return grad_out, None, None, None, None


def fake_quantize(x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor,
                  num_bits: int = 8, symmetric: bool = True) -> torch.Tensor:
    if symmetric:
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1
    return _FakeQuantSTE.apply(x, scale, zp, qmin, qmax)


class FakeQuantize(nn.Module):
    """Couples an observer with a fake-quant op.

    Two independent flags govern behavior:
        observe — if True, update the observer's running stats every forward.
        enabled — if True, apply fake-quant on the forward output.

    Typical lifecycle:
        1. observe=True, enabled=False during calibration.
        2. observe=True, enabled=True during QAT (stats keep refining).
        3. observe=False, enabled=True at eval / export.
    """

    def __init__(self, observer: nn.Module, num_bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.observer = observer
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.enabled: bool = False
        self.observe: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observe:
            self.observer(x)
        if not self.enabled:
            return x
        scale, zp = self.observer.scale_zp(self.num_bits, self.symmetric)
        # Broadcast per-channel scale/zp across input rank.
        if scale.dim() == 1 and x.dim() > 1:
            shape = [1] * x.dim()
            shape[0] = scale.numel()
            scale = scale.view(*shape)
            zp = zp.view(*shape)
        return fake_quantize(x, scale.to(x.dtype), zp.to(x.dtype),
                             self.num_bits, self.symmetric)
