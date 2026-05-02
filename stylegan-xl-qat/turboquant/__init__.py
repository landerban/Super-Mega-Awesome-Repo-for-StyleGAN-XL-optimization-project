"""TurboQuant: random Hadamard rotation followed by INT4 quantization.

Pipeline:
    1. Generate random sign vectors (one per quantized weight).
    2. Rotate weight by H = (1/sqrt(n)) * Hadamard @ diag(signs).
    3. Per-output-channel INT4 quantization of the rotated weight.
"""

from .hadamard import (
    hadamard_rotate, hadamard_unrotate, random_signs, make_block_signs, is_pow2,
)
from .rotate import rotate_linear_weight, rotate_conv_weight, collect_rotation_signs
from .compress import (
    CompressedWeight, compress_generator, reconstruct_weight,
    quantize_int4_per_channel, dequantize_int4_per_channel,
)

__all__ = [
    "hadamard_rotate", "hadamard_unrotate", "random_signs", "make_block_signs",
    "is_pow2",
    "rotate_linear_weight", "rotate_conv_weight", "collect_rotation_signs",
    "CompressedWeight", "compress_generator", "reconstruct_weight",
    "quantize_int4_per_channel", "dequantize_int4_per_channel",
]
