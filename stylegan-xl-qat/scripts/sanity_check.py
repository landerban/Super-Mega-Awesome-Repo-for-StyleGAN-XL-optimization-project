"""Load the pretrained StyleGAN-XL checkpoint and generate 4 images.

Verifies environment, checkpoint integrity, and that the synthesis network
can be imported and run on the configured device.
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Make sibling packages (qat/, eval/, ...) importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    sys.path.insert(0, "stylegan-xl")  # cloned StyleGAN-XL repo
    import dnnlib
    import legacy

    pkl = cfg.paths.pretrained_pkl
    print(f"Loading {pkl} ...")
    with dnnlib.util.open_url(pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to("cuda")
    G.eval()

    # 4 random ImageNet classes for variety.
    classes = torch.tensor([207, 248, 285, 933])  # golden retriever, husky, cat, lemon
    z = torch.randn(4, G.z_dim, device="cuda")
    c = torch.zeros(4, G.c_dim, device="cuda")
    c[torch.arange(4), classes] = 1.0

    with torch.no_grad():
        img = G(z, c, truncation_psi=1.0, noise_mode="const")

    out_dir = Path(cfg.paths.out_dir) / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = (img.clamp(-1, 1) + 1) / 2
    from torchvision.utils import save_image
    save_image(img, out_dir / "sanity.png", nrow=2)
    print(f"Saved 4 images to {out_dir / 'sanity.png'}")


if __name__ == "__main__":
    main()
