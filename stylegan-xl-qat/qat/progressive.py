"""Progressive quantization scheduler.

Toggles weight/activation quantization on each synthesis block based on the
current training iteration, per the schedule defined in the config.

The synthesis network is expected to expose blocks keyed by output
resolution: `synthesis.b4`, `synthesis.b8`, ..., `synthesis.b512`.
"""

from dataclasses import dataclass
from typing import Iterable, List, Dict

import torch.nn as nn

from .wrappers import QuantizedConvWrapper


@dataclass
class Stage:
    end: int                 # Iterations < end fall into this stage.
    w_blocks: List[int]      # Resolutions whose weights should be quantized.
    a_blocks: List[int]      # Resolutions whose activations should be quantized.


class ProgressiveQuantScheduler:
    def __init__(self, synthesis: nn.Module, stages: Iterable[Stage],
                 resolutions: Iterable[int] = (4, 8, 16, 32, 64, 128, 256, 512)):
        self.synthesis = synthesis
        self.stages: List[Stage] = sorted(stages, key=lambda s: s.end)
        self.resolutions = list(resolutions)

    def _block(self, res: int) -> nn.Module | None:
        return getattr(self.synthesis, f"b{res}", None)

    def _set_block(self, res: int, *, w: bool, a: bool) -> None:
        block = self._block(res)
        if block is None:
            return
        for m in block.modules():
            if isinstance(m, QuantizedConvWrapper):
                m.quant_weight = w
                m.quant_act = a

    def step(self, iteration: int) -> Dict[int, Dict[str, bool]]:
        """Update flags for the given iteration. Returns res -> {w, a} for logging."""
        stage = next((s for s in self.stages if iteration < s.end), self.stages[-1])
        flags: Dict[int, Dict[str, bool]] = {}
        for res in self.resolutions:
            w = res in stage.w_blocks
            a = res in stage.a_blocks
            self._set_block(res, w=w, a=a)
            flags[res] = {"w": w, "a": a}
        return flags

    @classmethod
    def from_config(cls, synthesis: nn.Module, cfg) -> "ProgressiveQuantScheduler":
        stages = [
            Stage(end=int(s["end"]),
                  w_blocks=list(s["w_blocks"]),
                  a_blocks=list(s["a_blocks"]))
            for s in cfg["stages"]
        ]
        resolutions = list(cfg.get("resolutions", (4, 8, 16, 32, 64, 128, 256, 512)))
        return cls(synthesis, stages, resolutions)
