"""Apply TurboQuant (Hadamard rotation + INT4) to a QAT-trained checkpoint.

Reads a converged QAT checkpoint, runs random Hadamard rotation per
synthesis weight tensor, then INT4-quantizes per-output-channel. Saves the
compressed dict + sign vectors needed for inference.
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--qat-ckpt", required=True,
                    help="Path to converged QAT checkpoint (qat_final.pt).")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    sys.path.insert(0, "stylegan-xl")
    import dnnlib
    import legacy

    with dnnlib.util.open_url(cfg.paths.pretrained_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to("cuda").eval()

    from qat.wrappers import wrap_synthesis_block
    for res in cfg.progressive.resolutions:
        block = getattr(G.synthesis, f"b{res}", None)
        if block is not None:
            wrap_synthesis_block(
                block,
                weight_bits=cfg.quant.weight_bits,
                act_bits=cfg.quant.act_bits,
                ema_momentum=cfg.quant.ema_momentum,
            )
    state = torch.load(args.qat_ckpt, map_location="cuda")
    G.load_state_dict(state["G"], strict=False)

    from turboquant.rotate import collect_rotation_signs
    from turboquant.compress import compress_generator

    signs = collect_rotation_signs(G.synthesis,
                                   block_size=cfg.turboquant.block_size,
                                   seed=cfg.turboquant.hadamard_seed)
    compressed = compress_generator(G.synthesis, signs)

    out = Path(cfg.paths.out_dir) / "turboquant.pt"
    torch.save({"compressed": compressed, "signs": signs}, out)

    int4_bytes = sum(c.qweight.numel() for c in compressed.values()) / 2
    scale_bytes = sum(c.scale.numel() * c.scale.element_size() for c in compressed.values())
    sign_bytes = sum(c.signs.numel() * c.signs.element_size() for c in compressed.values())
    total_mb = (int4_bytes + scale_bytes + sign_bytes) / (1024 ** 2)

    print(f"Saved TurboQuant-compressed weights to {out}")
    print(f"  layers compressed: {len(compressed)}")
    print(f"  INT4 storage:      {int4_bytes / (1024**2):.2f} MB")
    print(f"  scales:            {scale_bytes / (1024**2):.2f} MB")
    print(f"  signs:             {sign_bytes / (1024**2):.2f} MB")
    print(f"  total:             {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
