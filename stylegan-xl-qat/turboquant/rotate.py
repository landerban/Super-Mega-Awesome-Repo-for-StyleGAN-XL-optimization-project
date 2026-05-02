"""Apply random Hadamard rotations to weight matrices.

For a linear layer y = W x, we insert R^T R = I:
    y = W R^T R x  =>  y_rot = (W R^T) (R x).
The rotated weight (W R^T) is what we quantize. The matching R is applied to
input activations at inference time (or folded into the previous layer).
"""

from typing import Dict

import torch
import torch.nn as nn

from .hadamard import hadamard_rotate, make_block_signs


def rotate_linear_weight(weight: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Rotate a (out, in) linear weight along the input (last) axis."""
    return hadamard_rotate(weight, signs)


def rotate_conv_weight(weight: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """For Conv2d weight (out, in, kh, kw), rotate along the `in` axis only.

    We reshape (out, in, kh, kw) -> (out*kh*kw, in), rotate along the trailing
    `in` axis, then reshape back.
    """
    out_ch, in_ch, kh, kw = weight.shape
    w = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_ch)
    w_rot = hadamard_rotate(w, signs)
    return w_rot.view(out_ch, kh, kw, in_ch).permute(0, 3, 1, 2).contiguous()


def collect_rotation_signs(generator: nn.Module, block_size: int, seed: int = 0,
                           skip: tuple[str, ...] = ("mapping", "torgb", "ToRGB",
                                                    "affine"),
                           ) -> Dict[str, torch.Tensor]:
    """Generate one sign vector per eligible weight tensor in the synthesis net.

    Skips parameters whose names contain any of `skip` substrings (mapping,
    toRGB, affine style projections). Skips weights whose input dim isn't
    divisible by `block_size`.
    """
    signs: Dict[str, torch.Tensor] = {}
    for name, p in generator.named_parameters():
        if any(s in name for s in skip):
            continue
        if p.dim() not in (2, 4):
            continue
        in_dim = p.shape[1]
        bs = min(block_size, in_dim)
        if in_dim % bs != 0:
            continue
        # Stable per-tensor seed so the rotation is reproducible.
        per_seed = (seed * 0x9E3779B1) ^ (hash(name) & 0xFFFFFFFF)
        signs[name] = make_block_signs(
            in_dim, bs, seed=per_seed & 0x7FFFFFFF,
            device=p.device, dtype=p.dtype,
        )
    return signs
