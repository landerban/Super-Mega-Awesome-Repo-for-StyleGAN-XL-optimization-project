"""Run calibration to populate observer statistics before QAT.

Wraps every synthesis block with QuantizedConvWrapper, then forward-passes
`cfg.calibration.num_batches` random latents to fill in min/max stats.
Saves the calibrated state_dict so train_qat.py can load it as a warm start.
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
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    sys.path.insert(0, "stylegan-xl")
    import dnnlib
    import legacy

    from qat.wrappers import wrap_synthesis_block
    from qat.calibration import calibrate

    with dnnlib.util.open_url(cfg.paths.pretrained_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to("cuda").eval()

    for res in cfg.progressive.resolutions:
        block = getattr(G.synthesis, f"b{res}", None)
        if block is not None:
            wrap_synthesis_block(
                block,
                weight_bits=cfg.quant.weight_bits,
                act_bits=cfg.quant.act_bits,
                ema_momentum=cfg.quant.ema_momentum,
            )

    def batches():
        bs = cfg.calibration.batch_size
        for _ in range(cfg.calibration.num_batches):
            z = torch.randn(bs, G.z_dim, device="cuda")
            c = torch.zeros(bs, G.c_dim, device="cuda")
            c[torch.arange(bs), torch.randint(0, G.c_dim, (bs,))] = 1.0
            yield z, c

    calibrate(G, batches(), num_batches=cfg.calibration.num_batches)

    out = Path(cfg.paths.calibration_cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), out)
    print(f"Saved calibrated state_dict to {out}")


if __name__ == "__main__":
    main()
