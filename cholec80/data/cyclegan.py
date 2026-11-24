"""Unaligned image-pair dataset for CycleGAN (domain A <-> domain B)."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(directory: str | Path) -> list[str]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no images found in {directory}")
    return [str(p) for p in files]


class UnalignedImageDataset(Dataset):
    """Random pairs ``(A, B)``; length = max(len(A), len(B)) as in CycleGAN."""

    def __init__(self, a_dir: str | Path, b_dir: str | Path, transform, unaligned: bool = True):
        self.files_a = _list_images(a_dir)
        self.files_b = _list_images(b_dir)
        self.transform = transform
        self.unaligned = unaligned

    def __len__(self) -> int:
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, index: int) -> dict:
        item_a = self.transform(Image.open(self.files_a[index % len(self.files_a)]).convert("RGB"))
        if self.unaligned:
            j = random.randint(0, len(self.files_b) - 1)
        else:
            j = index % len(self.files_b)
        item_b = self.transform(Image.open(self.files_b[j]).convert("RGB"))
        return {"A": item_a, "B": item_b}
