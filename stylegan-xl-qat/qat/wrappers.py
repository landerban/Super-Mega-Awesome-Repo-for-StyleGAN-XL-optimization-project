"""Wrappers that splice fake-quantization into StyleGAN-XL synthesis modules.

StyleGAN-XL synthesis blocks (`networks_stylegan3.py`) expose modulated
convolutions named `conv0` and `conv1`, plus a `torgb` head. We wrap the
two convs and leave `torgb` and the mapping network untouched per the
methodology.

NOTE on modulated convs: StyleGAN's modulated conv multiplies the weight by
a per-sample style scalar inside the layer's forward. Our wrapper applies
fake-quant to the *unmodulated* base weight before the layer reads it. If
you want to fake-quant the *post-modulation* weight instead, hook inside
the modulated_conv2d call — see TODO below.
"""

import torch
from torch import nn

from .observers import PerChannelMinMaxObserver, PerTensorEMAMinMaxObserver
from .fake_quant import FakeQuantize


class QuantizedConvWrapper(nn.Module):
    """Wraps a conv-like inner module so that its `weight` and inputs are
    fake-quantized when the corresponding flags are enabled.

    The inner module's `weight` Parameter is swapped for a fake-quantized
    view on each forward pass and restored afterwards, which keeps the
    underlying optimizer state pointing at the real (FP) parameter.
    """

    def __init__(self, inner: nn.Module, weight_bits: int = 8, act_bits: int = 8,
                 ema_momentum: float = 0.99):
        super().__init__()
        self.inner = inner
        out_ch = inner.weight.shape[0]
        self.w_fq = FakeQuantize(
            PerChannelMinMaxObserver(out_ch, ch_axis=0),
            num_bits=weight_bits, symmetric=True,
        )
        self.a_fq = FakeQuantize(
            PerTensorEMAMinMaxObserver(ema_momentum),
            num_bits=act_bits, symmetric=False,
        )

    @property
    def quant_weight(self) -> bool:
        return self.w_fq.enabled

    @quant_weight.setter
    def quant_weight(self, v: bool) -> None:
        self.w_fq.enabled = bool(v)

    @property
    def quant_act(self) -> bool:
        return self.a_fq.enabled

    @quant_act.setter
    def quant_act(self, v: bool) -> None:
        self.a_fq.enabled = bool(v)

    def forward(self, x, *args, **kwargs):
        x = self.a_fq(x)
        orig_w = self.inner.weight
        # TODO: For modulated convs, hook inside the modulated_conv2d call
        # rather than swapping the param, so we fake-quant the post-style
        # weight. The current swap is correct only for vanilla conv layers.
        fq_w = self.w_fq(orig_w)
        self.inner.weight = nn.Parameter(fq_w, requires_grad=orig_w.requires_grad)
        try:
            out = self.inner(x, *args, **kwargs)
        finally:
            self.inner.weight = orig_w
        return out


def wrap_synthesis_block(block: nn.Module, weight_bits: int = 8, act_bits: int = 8,
                         ema_momentum: float = 0.99) -> nn.Module:
    """Wrap the conv layers inside a StyleGAN-XL synthesis block in-place.

    Targets attributes named `conv0`, `conv1`. The toRGB layer is intentionally
    left untouched.
    """
    for name in ("conv0", "conv1"):
        inner = getattr(block, name, None)
        if inner is None or isinstance(inner, QuantizedConvWrapper):
            continue
        setattr(block, name, QuantizedConvWrapper(
            inner, weight_bits=weight_bits, act_bits=act_bits,
            ema_momentum=ema_momentum,
        ))
    return block


def iter_quant_wrappers(module: nn.Module):
    """Yield every QuantizedConvWrapper inside `module`."""
    for m in module.modules():
        if isinstance(m, QuantizedConvWrapper):
            yield m
