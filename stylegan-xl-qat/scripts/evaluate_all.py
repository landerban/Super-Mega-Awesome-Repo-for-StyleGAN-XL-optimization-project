"""Evaluate every variant: FP32 / +BF16 / +QAT / +QAT+TQ.

Loads each variant in turn, computes FID + latency + size, and writes one
JSON file with the comparison.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_pretrained(cfg):
    sys.path.insert(0, "stylegan-xl")
    import dnnlib
    import legacy
    with dnnlib.util.open_url(cfg.paths.pretrained_pkl) as f:
        return legacy.load_network_pkl(f)["G_ema"].to("cuda").eval()


def load_fp32(cfg):
    return _load_pretrained(cfg)


def load_bf16(cfg):
    return _load_pretrained(cfg).to(torch.bfloat16)


def load_qat(cfg, ckpt_path):
    G = _load_pretrained(cfg)
    from qat.wrappers import wrap_synthesis_block
    for res in cfg.progressive.resolutions:
        block = getattr(G.synthesis, f"b{res}", None)
        if block is not None:
            wrap_synthesis_block(block,
                                 weight_bits=cfg.quant.weight_bits,
                                 act_bits=cfg.quant.act_bits,
                                 ema_momentum=cfg.quant.ema_momentum)
    state = torch.load(ckpt_path, map_location="cuda")
    G.load_state_dict(state["G"], strict=False)
    # Force all FakeQuantize modules into eval mode (enabled=True, observe=False).
    from qat.calibration import set_observe, set_enabled
    set_observe(G, False)
    set_enabled(G, True)
    return G


def load_qat_tq(cfg, qat_ckpt, tq_ckpt):
    """Load QAT + replace synthesis weights with the rotated/dequantized
    INT4 reconstructions. Inference still happens in FP - we're measuring
    the *quality* loss from INT4, not the latency win from kernel-level INT4.
    """
    G = load_qat(cfg, qat_ckpt)
    tq = torch.load(tq_ckpt, map_location="cuda")
    from turboquant.compress import reconstruct_weight
    sd = G.synthesis.state_dict()
    for name, cw in tq["compressed"].items():
        if name in sd:
            sd[name] = reconstruct_weight(cw)
    G.synthesis.load_state_dict(sd, strict=False)
    return G


def evaluate(name: str, G, cfg) -> dict:
    from eval.fid import compute_fid
    from eval.latency import measure_latency
    from eval.memory import model_size_bytes
    return {
        "variant": name,
        "fid": compute_fid(G, cfg.paths.dataset_zip,
                           num_samples=cfg.eval.num_fid_samples,
                           batch_size=cfg.eval.fid_batch_size),
        "latency": measure_latency(G, G.z_dim, G.c_dim, batch_size=1),
        "size": model_size_bytes(G),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--qat-ckpt", required=True)
    ap.add_argument("--tq-ckpt", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    results = []
    variants = [
        ("fp32", lambda: load_fp32(cfg)),
        ("bf16", lambda: load_bf16(cfg)),
        ("qat", lambda: load_qat(cfg, args.qat_ckpt)),
        ("qat_tq", lambda: load_qat_tq(cfg, args.qat_ckpt, args.tq_ckpt)),
    ]
    for name, loader in variants:
        print(f"\n=== Evaluating {name} ===")
        G = loader()
        results.append(evaluate(name, G, cfg))
        del G
        torch.cuda.empty_cache()

    out = Path(cfg.paths.out_dir) / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
