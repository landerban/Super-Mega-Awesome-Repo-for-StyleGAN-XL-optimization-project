"""Post-rotation INT4 quantization of generator weights."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .rotate import rotate_linear_weight, rotate_conv_weight


@dataclass
class CompressedWeight:
    qweight: torch.Tensor       # int8 storage holding signed INT4 values.
    scale: torch.Tensor         # per-output-channel scale.
    signs: torch.Tensor         # rotation signs (length = in_dim).
    shape: torch.Size           # original weight shape.
    is_conv: bool


def quantize_int4_per_channel(w: torch.Tensor, symmetric: bool = True
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel INT4 quantization.

    Returns (qweight, scale). qweight is stored in int8 with values in [-8, 7].
    """
    if not symmetric:
        raise NotImplementedError("Only symmetric INT4 supported in v1.")
    qmax = 7
    flat = w.reshape(w.shape[0], -1)
    amax = flat.abs().max(dim=1).values
    scale = (amax / qmax).clamp_min(1e-8)
    q = torch.round(flat / scale.unsqueeze(1)).clamp(-qmax - 1, qmax)
    return q.to(torch.int8).view_as(w), scale


def dequantize_int4_per_channel(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    flat = q.to(scale.dtype).reshape(q.shape[0], -1) * scale.unsqueeze(1)
    return flat.view_as(q)


def compress_generator(generator: nn.Module, signs: Dict[str, torch.Tensor],
                       skip: tuple[str, ...] = ("mapping", "torgb", "ToRGB", "affine"),
                       ) -> Dict[str, CompressedWeight]:
    """Rotate then INT4-quantize all eligible weight tensors in `generator`.

    Returns a dict keyed by parameter name. Mapping/toRGB are skipped. Only
    parameters present in `signs` are processed (so non-power-of-two inputs
    that the sign-collector dropped are also dropped here).
    """
    compressed: Dict[str, CompressedWeight] = {}
    for name, p in generator.named_parameters():
        if any(s in name for s in skip):
            continue
        if name not in signs:
            continue
        w = p.data
        if w.dim() == 4:
            w_rot = rotate_conv_weight(w, signs[name])
            is_conv = True
        elif w.dim() == 2:
            w_rot = rotate_linear_weight(w, signs[name])
            is_conv = False
        else:
            continue
        q, scale = quantize_int4_per_channel(w_rot)
        compressed[name] = CompressedWeight(
            qweight=q, scale=scale, signs=signs[name],
            shape=w.shape, is_conv=is_conv,
        )
    return compressed


def reconstruct_weight(cw: CompressedWeight) -> torch.Tensor:
    """Rebuild a (rotated, dequantized) weight tensor from its INT4 form."""
    return dequantize_int4_per_channel(cw.qweight, cw.scale)
