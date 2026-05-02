"""Wall-clock inference timing using CUDA events."""

import torch


@torch.no_grad()
def measure_latency(generator, z_dim: int, num_classes: int, *,
                    batch_size: int = 1, warmup: int = 20, repeats: int = 100,
                    device: str = "cuda") -> dict:
    """Time `generator(z, c)` over `repeats` runs after `warmup`.

    Returns mean / p50 / p95 / p99 ms and throughput (imgs/s).
    """
    generator.eval()
    z = torch.randn(batch_size, z_dim, device=device)
    c = torch.zeros(batch_size, num_classes, device=device)
    c[:, 0] = 1.0  # arbitrary class

    for _ in range(warmup):
        _ = generator(z, c)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        _ = generator(z, c)
        ends[i].record()
    torch.cuda.synchronize()

    times = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return {
        "mean_ms": times.mean().item(),
        "p50_ms": times.median().item(),
        "p95_ms": times.quantile(0.95).item(),
        "p99_ms": times.quantile(0.99).item(),
        "throughput_imgs_per_s": batch_size * 1000.0 / times.mean().item(),
        "batch_size": batch_size,
    }
