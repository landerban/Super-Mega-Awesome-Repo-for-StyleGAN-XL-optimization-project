"""Compute FID / latency / size on the pretrained FP32 generator.

This is the reference number against which all quantized variants are
compared. Run once, then re-use the JSON output throughout the project.
"""

import argparse
import json
import sys
from pathlib import Path

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

    with dnnlib.util.open_url(cfg.paths.pretrained_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to("cuda").eval()

    from eval.fid import compute_fid
    from eval.latency import measure_latency
    from eval.memory import model_size_bytes

    fid = compute_fid(G, cfg.paths.dataset_zip,
                      num_samples=cfg.eval.num_fid_samples,
                      batch_size=cfg.eval.fid_batch_size)
    lat = measure_latency(G, G.z_dim, G.c_dim, batch_size=1)
    sz = model_size_bytes(G)

    out = {"variant": "fp32_baseline", "fid": fid, "latency": lat, "size": sz}
    out_path = Path(cfg.paths.out_dir) / "baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
