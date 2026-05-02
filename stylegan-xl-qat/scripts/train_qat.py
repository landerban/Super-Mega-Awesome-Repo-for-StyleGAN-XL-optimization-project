"""Main QAT training loop with progressive scheduling.

Loads the pretrained StyleGAN-XL G_ema as the starting point for fine-tuning,
wraps synthesis blocks for fake-quant, freezes the discriminator, and runs
~20k iters of progressive QAT under bf16 autocast with 8-bit Adam and
gradient checkpointing.

Loss: non-saturating logistic on the (frozen) discriminator. We do NOT
update D - this is a generator-only fine-tune. The intent is to recover
quality lost to weight/activation quantization, not to keep training a GAN.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import bitsandbytes as bnb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def setup_models(cfg):
    """Load G_ema (treated as starting weights) and D from the pretrained pkl."""
    sys.path.insert(0, "stylegan-xl")
    import dnnlib
    import legacy
    with dnnlib.util.open_url(cfg.paths.pretrained_pkl) as f:
        ckpt = legacy.load_network_pkl(f)
    G = ckpt["G_ema"].to("cuda").train()
    D = ckpt["D"].to("cuda").eval()
    for p in D.parameters():
        p.requires_grad_(False)
    return G, D


def wrap_for_qat(G, cfg) -> None:
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


def enable_grad_checkpoint(G, cfg) -> None:
    """Wrap each synthesis block's forward in `checkpoint(...)` to save VRAM.

    Boundary granularity = one synthesis block, per the methodology spec.
    """
    from torch.utils.checkpoint import checkpoint
    for res in cfg.progressive.resolutions:
        block = getattr(G.synthesis, f"b{res}", None)
        if block is None:
            continue
        orig_forward = block.forward

        def make_ckpt_forward(f):
            def _fwd(*args, **kwargs):
                return checkpoint(f, *args, use_reentrant=False, **kwargs)
            return _fwd

        block.forward = make_ckpt_forward(orig_forward)


@torch.no_grad()
def update_ema(ema_params, src_params, decay: float) -> None:
    for ep, sp in zip(ema_params, src_params):
        ep.mul_(decay).add_(sp.detach().to(ep.dtype), alpha=1 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", default=None,
                    help="Path to a previous QAT checkpoint to resume from.")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    torch.manual_seed(cfg.project.seed)
    device = "cuda"

    G, D = setup_models(cfg)
    wrap_for_qat(G, cfg)

    # Warm-start observer stats from calibration pass if present.
    calib = Path(cfg.paths.calibration_cache)
    if calib.exists() and args.resume is None:
        print(f"Loading calibrated state from {calib}")
        G.load_state_dict(torch.load(calib, map_location=device), strict=False)

    if cfg.train.grad_checkpoint:
        enable_grad_checkpoint(G, cfg)

    from qat.progressive import ProgressiveQuantScheduler
    scheduler = ProgressiveQuantScheduler.from_config(G.synthesis, cfg.progressive)

    g_params = [p for p in G.parameters() if p.requires_grad]
    if cfg.train.use_8bit_adam:
        opt = bnb.optim.Adam8bit(
            g_params, lr=cfg.train.lr_g,
            betas=(cfg.train.beta1, cfg.train.beta2), eps=cfg.train.eps,
        )
    else:
        opt = torch.optim.Adam(
            g_params, lr=cfg.train.lr_g,
            betas=(cfg.train.beta1, cfg.train.beta2), eps=cfg.train.eps,
        )

    # FP16 EMA shadow (cheap to keep on GPU).
    ema_dtype = getattr(torch, cfg.train.ema_dtype)
    ema_params = [p.detach().clone().to(ema_dtype) for p in g_params]

    iteration_start = 0
    if args.resume is not None:
        from utils.checkpoints import load_checkpoint
        state = load_checkpoint(args.resume, G=G, opt=opt, map_location=device)
        iteration_start = int(state.get("iter", 0))
        for ep, sp in zip(ema_params, state["ema"]):
            ep.copy_(sp.to(ep.device, dtype=ep.dtype))

    from data.dataset import ImageNetZipDataset
    from data.augment import build_augment_pipeline
    ds = ImageNetZipDataset(cfg.paths.dataset_zip,
                            resolution=cfg.model.resolution,
                            use_turbojpeg=cfg.data.use_turbojpeg)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor, persistent_workers=True,
    )
    augment = build_augment_pipeline(**cfg.data.augment).to(device)

    from utils.logging import init_wandb, log_metrics, finish
    run = init_wandb(cfg)

    G.train()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                  enabled=cfg.train.bf16)
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iteration = iteration_start
    data_iter = iter(loader)
    t0 = time.time()

    while iteration < cfg.progressive.total_iters:
        flags = scheduler.step(iteration)

        try:
            real_img, real_c = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_img, real_c = next(data_iter)
        real_img = real_img.to(device, non_blocking=True)
        real_c = real_c.to(device, non_blocking=True)
        real_c_oh = F.one_hot(real_c, num_classes=cfg.model.num_classes).to(real_img.dtype)
        real_img = augment(real_img)

        z = torch.randn(real_img.size(0), G.z_dim, device=device)

        opt.zero_grad(set_to_none=True)
        with autocast:
            fake_img = G(z, real_c_oh)
            d_fake = D(fake_img, real_c_oh)
            # Non-saturating logistic loss for the generator. D is frozen.
            g_loss = F.softplus(-d_fake).mean()

        g_loss.backward()
        opt.step()

        if iteration % cfg.train.ema_steps == 0:
            update_ema(ema_params, g_params, cfg.train.ema_decay)

        if iteration % cfg.train.log_every == 0:
            metrics = {
                "iter": iteration,
                "g_loss": g_loss.item(),
                "iters_per_s": (iteration - iteration_start + 1) / max(time.time() - t0, 1e-6),
            }
            metrics.update({f"q/b{r}/w": int(v["w"]) for r, v in flags.items()})
            metrics.update({f"q/b{r}/a": int(v["a"]) for r, v in flags.items()})
            log_metrics(run, metrics)

        if iteration > 0 and iteration % cfg.train.ckpt_every == 0:
            from utils.checkpoints import save_checkpoint
            save_checkpoint(out_dir / f"qat_{iteration:07d}.pt",
                            G=G, ema=ema_params, opt=opt, iteration=iteration)

        iteration += 1

    from utils.checkpoints import save_checkpoint
    save_checkpoint(out_dir / "qat_final.pt", G=G, ema=ema_params, opt=opt,
                    iteration=iteration)
    finish(run)
    print(f"Done. Final checkpoint at {out_dir / 'qat_final.pt'}")


if __name__ == "__main__":
    main()
