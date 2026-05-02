"""Random Hadamard rotation utilities.

Wraps `fast-hadamard-transform`. The rotation we apply is
    H = (1/sqrt(n)) * Hadamard_n @ diag(s)    s_i in {-1, +1}
which is orthogonal, distributes weight magnitude across the rotated axis,
and is its own inverse up to the 1/sqrt(n) scale.
"""

from typing import Optional

import torch

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def _require_lib() -> None:
    if hadamard_transform is None:
        raise ImportError(
            "fast-hadamard-transform is required for TurboQuant. "
            "Install with `pip install fast-hadamard-transform`."
        )


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def random_signs(dim: int, generator: Optional[torch.Generator] = None,
                 device: str = "cuda", dtype: torch.dtype = torch.float32
                 ) -> torch.Tensor:
    s = torch.randint(0, 2, (dim,), generator=generator, device=device, dtype=torch.int8)
    return s.to(dtype) * 2 - 1


def hadamard_rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Apply H @ x along the last axis where H = (1/sqrt(n)) Hadamard @ diag(signs)."""
    _require_lib()
    if not is_pow2(x.shape[-1]):
        raise ValueError(f"Hadamard rotation requires power-of-two last dim; got {x.shape[-1]}")
    return hadamard_transform(x * signs, scale=1.0 / (x.shape[-1] ** 0.5))


def hadamard_unrotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Inverse of hadamard_rotate (Hadamard is self-inverse up to scale)."""
    _require_lib()
    return hadamard_transform(x, scale=1.0 / (x.shape[-1] ** 0.5)) * signs


def make_block_signs(dim: int, block_size: int, seed: int = 0,
                     device: str = "cuda", dtype: torch.dtype = torch.float32
                     ) -> torch.Tensor:
    """For dims that aren't power-of-two, rotate within blocks of `block_size`.
    Returns signs of shape (dim,) where each block has its own random pattern."""
    if dim % block_size != 0:
        raise ValueError(f"dim {dim} must be divisible by block_size {block_size}")
    if not is_pow2(block_size):
        raise ValueError(f"block_size must be power of two; got {block_size}")
    g = torch.Generator(device=device).manual_seed(seed)
    return random_signs(dim, generator=g, device=device, dtype=dtype)
