"""Dataset reader for StyleGAN-XL's zip format, with PyTurboJPEG decoding.

Layout assumed (matches StyleGAN-XL's `dataset_tool.py` output):
    dataset.zip/
        00000/img00000000.jpg
        ...
        dataset.json   -> {"labels": [["00000/img00000000.jpg", 0], ...]}

PyTurboJPEG decodes ~3-5x faster than PIL on JPEGs; we fall back to PIL if
the binding isn't available (so the dataset still works in dev environments).
"""

import io
import json
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _TJ = TurboJPEG()
except Exception:
    _TJ = None
    TJPF_RGB = None

try:
    from PIL import Image
except Exception:
    Image = None


class ImageNetZipDataset(Dataset):
    def __init__(self, zip_path: str | Path, resolution: int = 512,
                 use_turbojpeg: bool = True):
        self.zip_path = str(zip_path)
        self.resolution = resolution
        self.use_turbojpeg = use_turbojpeg and _TJ is not None
        self._zf: zipfile.ZipFile | None = None
        with zipfile.ZipFile(self.zip_path) as zf:
            with zf.open("dataset.json") as f:
                meta = json.load(f)
        self.labels = meta["labels"]

    def _open(self) -> zipfile.ZipFile:
        # Each worker opens its own handle lazily.
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel, label = self.labels[idx]
        with self._open().open(rel) as f:
            buf = f.read()

        if self.use_turbojpeg:
            arr = _TJ.decode(buf, pixel_format=TJPF_RGB)
        else:
            if Image is None:
                raise RuntimeError("Neither PyTurboJPEG nor PIL is available.")
            arr = np.array(Image.open(io.BytesIO(buf)).convert("RGB"))

        if arr.shape[0] != self.resolution or arr.shape[1] != self.resolution:
            raise ValueError(
                f"Expected {self.resolution}x{self.resolution}; got {arr.shape}. "
                "Re-run dataset_tool.py with the right --resolution."
            )

        # HWC uint8 -> CHW float in [-1, 1] (StyleGAN convention).
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        x = x.to(torch.float32).mul_(1.0 / 127.5).sub_(1.0)
        return x, int(label)
