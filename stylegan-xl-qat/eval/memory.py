"""Peak VRAM and parameter-byte accounting.

`model_size_bytes` reports *raw* parameter storage by dtype. INT4 is not a
native PyTorch dtype, so the int4 bucket is populated externally (e.g. by
counting CompressedWeight.qweight elements at 4 bits each).
"""

import torch
import torch.nn as nn


def model_size_bytes(model: nn.Module) -> dict:
    sizes = {"fp32": 0, "fp16": 0, "bf16": 0, "int8": 0, "other": 0}
    for p in model.parameters():
        n = p.numel()
        if p.dtype == torch.float32:
            sizes["fp32"] += 4 * n
        elif p.dtype == torch.float16:
            sizes["fp16"] += 2 * n
        elif p.dtype == torch.bfloat16:
            sizes["bf16"] += 2 * n
        elif p.dtype == torch.int8:
            sizes["int8"] += n
        else:
            sizes["other"] += p.element_size() * n
    sizes["total_mb"] = sum(sizes.values()) / (1024 ** 2)
    return sizes


@torch.no_grad()
def measure_peak_vram(fn, *args, device: str = "cuda", **kwargs):
    """Run `fn(*args, **kwargs)` and return (result, peak_vram_mb)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    out = fn(*args, **kwargs)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return out, peak_bytes / (1024 ** 2)
