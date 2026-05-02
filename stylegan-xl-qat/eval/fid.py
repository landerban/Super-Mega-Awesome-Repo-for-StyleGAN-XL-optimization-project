"""FID computation — delegates to StyleGAN-XL's existing metrics module.

Assumes the cloned `stylegan-xl/` repo is on PYTHONPATH (it ships a
`metrics/` package with `metric_main.calc_metric` etc.).
"""

from pathlib import Path

import torch


def compute_fid(generator, dataset_zip: str | Path, *, num_samples: int = 50000,
                batch_size: int = 32, device: str = "cuda") -> float:
    """Compute fid50k_full against the dataset stored at `dataset_zip`."""
    try:
        from metrics import metric_main, metric_utils
    except ImportError as e:
        raise ImportError(
            "StyleGAN-XL's `metrics` module is not on PYTHONPATH. "
            "Insert the cloned stylegan-xl repo into sys.path first."
        ) from e

    opts = metric_utils.MetricOptions(
        G=generator,
        dataset_kwargs=dict(class_name="training.dataset.ImageFolderDataset",
                            path=str(dataset_zip)),
        num_gpus=1, rank=0, device=torch.device(device),
    )
    result = metric_main.calc_metric("fid50k_full", **vars(opts))
    return float(result["results"]["fid50k_full"])
