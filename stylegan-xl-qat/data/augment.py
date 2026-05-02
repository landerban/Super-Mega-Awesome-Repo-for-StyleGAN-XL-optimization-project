"""GPU-side augmentations using kornia.

Applied to a real-image batch already on device. Class-conditional safe —
we don't do any pair-mixing across samples (no MixUp / CutMix here).
"""

import torch.nn as nn

try:
    import kornia.augmentation as K
except ImportError:
    K = None


def build_augment_pipeline(hflip: bool = True, color_jitter: float = 0.0
                           ) -> nn.Module:
    if K is None:
        raise ImportError("kornia is required for GPU augmentations.")
    ops: list[nn.Module] = []
    if hflip:
        ops.append(K.RandomHorizontalFlip(p=0.5))
    if color_jitter > 0:
        ops.append(K.ColorJitter(color_jitter, color_jitter, color_jitter, 0))
    return nn.Sequential(*ops) if ops else nn.Identity()
