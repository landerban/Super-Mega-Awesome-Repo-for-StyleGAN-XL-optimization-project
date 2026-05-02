"""Calibration: populate observer statistics on a small subset of forwards
before turning fake-quant on.

During calibration we want `observe=True, enabled=False` so the observers
record stats but the network behaves identically to FP. After calibration,
the QAT loop flips `enabled=True` per the progressive schedule.
"""

from typing import Iterable

import torch
import torch.nn as nn

from .fake_quant import FakeQuantize


def set_observe(model: nn.Module, observe: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.observe = observe


def set_enabled(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.enabled = enabled


@torch.no_grad()
def calibrate(generator: nn.Module, batches: Iterable, *,
              device: str = "cuda", num_batches: int = 64) -> None:
    """Run forward passes through `generator(z, c)` to populate observer stats.

    `batches` should yield (z, c) tuples (latent, one-hot class).
    """
    generator.eval()
    set_observe(generator, True)
    set_enabled(generator, False)

    seen = 0
    for z, c in batches:
        z = z.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        _ = generator(z, c)
        seen += 1
        if seen >= num_batches:
            break

    # Freeze the observers post-calibration. The QAT loop can re-enable them
    # if we want stats to keep refining during training.
    set_observe(generator, False)
