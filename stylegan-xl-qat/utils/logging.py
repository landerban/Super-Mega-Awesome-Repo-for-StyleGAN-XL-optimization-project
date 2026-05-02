"""Thin WandB integration. No-ops gracefully if wandb isn't installed."""

from typing import Any

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(cfg) -> Any:
    if wandb is None:
        print("[logging] wandb not installed - logging to stdout only.")
        return None
    return wandb.init(
        project=cfg.project.name,
        config=dict(cfg) if not isinstance(cfg, dict) else cfg,
        resume="allow",
    )


def log_metrics(run, metrics: dict) -> None:
    if run is None:
        # Print compactly so we still have something during dry runs.
        kv = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in metrics.items())
        print(f"[log] {kv}")
        return
    run.log(metrics)


def finish(run) -> None:
    if run is None:
        return
    run.finish()
