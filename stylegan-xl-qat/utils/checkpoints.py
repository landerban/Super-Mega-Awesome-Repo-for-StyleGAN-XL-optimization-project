"""Checkpoint save/load preserving QAT state.

`G.state_dict()` already includes observer buffers (min_val, max_val,
initialized) because they're registered buffers on the FakeQuantize modules.
Fake-quant flags (`enabled`, `observe`) are plain Python attrs - we save
them in a sidecar dict so they round-trip across processes.
"""

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

from qat.fake_quant import FakeQuantize
from qat.wrappers import iter_quant_wrappers


def _quant_flags(G: nn.Module) -> dict:
    flags = {}
    for name, m in G.named_modules():
        if isinstance(m, FakeQuantize):
            flags[name] = {"enabled": m.enabled, "observe": m.observe}
    return flags


def _restore_quant_flags(G: nn.Module, flags: dict) -> None:
    for name, m in G.named_modules():
        if isinstance(m, FakeQuantize) and name in flags:
            m.enabled = bool(flags[name]["enabled"])
            m.observe = bool(flags[name]["observe"])


def save_checkpoint(path: str | Path, *, G: nn.Module,
                    ema: Iterable[torch.Tensor],
                    opt: torch.optim.Optimizer, iteration: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "G": G.state_dict(),
        "quant_flags": _quant_flags(G),
        "ema": [p.detach().cpu() for p in ema],
        "opt": opt.state_dict(),
        "iter": int(iteration),
    }, path)


def load_checkpoint(path: str | Path, *, G: nn.Module,
                    opt: Optional[torch.optim.Optimizer] = None,
                    map_location: str = "cuda") -> dict:
    state = torch.load(path, map_location=map_location)
    G.load_state_dict(state["G"], strict=False)
    if "quant_flags" in state:
        _restore_quant_flags(G, state["quant_flags"])
    if opt is not None and "opt" in state:
        opt.load_state_dict(state["opt"])
    return state
