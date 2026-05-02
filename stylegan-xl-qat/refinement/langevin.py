"""Discriminator-guided Langevin refinement (skeleton — v2 feature).

Treats D(x, c) as an unnormalized log-density and refines a sampled image
via overdamped Langevin dynamics:

    x_{t+1} = x_t + (eta/2) * grad_x D(x_t, c) + sqrt(eta) * sigma * noise

The refined sample stays in the support of natural images defined by D.

Status: skeleton only. Implement after the QAT + TurboQuant pipeline is
producing acceptable FIDs end-to-end.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LangevinConfig:
    steps: int = 10
    step_size: float = 0.01
    noise_scale: float = 0.005
    clamp: float | None = 1.0  # If set, clamp x to [-clamp, clamp] each step.


@torch.enable_grad()
def langevin_refine(x: torch.Tensor, c: torch.Tensor, discriminator: nn.Module,
                    cfg: LangevinConfig) -> torch.Tensor:
    """Refine `x` for `cfg.steps` Langevin updates using D as approximate EBM.

    NOTE: skeleton only. Before relying on this:
      - check D's logit sign convention (real → +) on this checkpoint.
      - tune step_size / noise_scale; values in cfg are placeholders.
      - decide whether to detach D (we likely want grads only on x).
      - consider truncation / projection back into the data manifold.
    """
    raise NotImplementedError(
        "Langevin refinement is a v2 feature — skip in v1."
    )
