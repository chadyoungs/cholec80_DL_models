"""Dataset for sim-to-real texture translation (domain A = simulation).

DIFFERENCES VS. A PLAIN CycleGAN DATASET
========================================
A vanilla CycleGAN dataset yields random (A, B) pairs and nothing else
(omit ``mask_dir`` and ``temporal`` here and you get exactly that).
For sim-to-real we need three extra things, each with a reason:

1. A PROTECTION MASK for every A frame (optional).
   The simulator knows exactly where instruments / semantically-labelled
   blood decals are.  We load that mask alongside the frame so the training
   loop can composite the original pixels back over the translation
   (see ``cholec80_dl.sim2real.blend_protected``).

2. A TEMPORAL PAIR (A_t, A_{t+1}) instead of a single A frame (optional).
   The temporal-consistency loss needs two consecutive frames of the SAME
   video.  Frames are stored as ``<video>/<frame:06d>.jpg`` and sorted by
   name, so "index + 1" is the next frame *iff* it belongs to the same
   video folder.  A ``pair_valid`` flag marks the last frame of each video
   (and singleton folders) where no next frame exists.

3. SHARED GEOMETRIC AUGMENTATION.
   Random crop / flip must be applied with the SAME random parameters to
   the image, its mask and the next frame -- otherwise the mask no longer
   aligns with the instruments and the flow no longer aligns with the
   motion.  This is why we call ``torchvision.transforms.functional`` with
   explicitly drawn parameters instead of stacking random transforms.
   Domain B gets its own independent augmentation (it is unpaired anyway).

MASK FILE CONVENTION
====================
``mask_dir`` mirrors the relative layout of ``a_dir`` with ``.png`` files:

    a_dir/video01/000025.jpg   ->   mask_dir/video01/000025.png

White (255) = protect this pixel, black (0) = free to translate.
Missing mask files are treated as "no protection" (all zeros), so you can
start without masks and add them later for the frames that matter.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images_recursive(directory: str | Path) -> list[Path]:
    """Sorted image list, recursing into per-video subfolders.

    Sorting by the full relative path gives frame order inside each video
    because frames are zero-padded (``000025.jpg``).  NEVER sort by string
    without zero padding -- '1000' < '100' lexically, a bug the legacy
    SlowFast loader actually had.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no images found in {directory}")
    return files


class Sim2RealDataset(Dataset):
    """Unpaired (sim, real) samples with optional mask + temporal pair.

    Each item is a dict of tensors:
        A          (3,H,W) float in [-1,1]   simulated frame at time t
        B          (3,H,W) float in [-1,1]   random real frame (unpaired)
        mask_a     (1,H,W) float in {0,1}    1 = protect (keep original)
        A_next     (3,H,W) float in [-1,1]   frame t+1 (== A if no pair)
        pair_valid ()      bool              True if A_next really is t+1

    Length is max(len(A), len(B)) -- the CycleGAN convention: one "epoch"
    sweeps the larger domain once, cycling the smaller one.
    """

    def __init__(
        self,
        a_dir: str | Path,
        b_dir: str | Path,
        size: int = 256,
        train: bool = True,
        mask_dir: str | Path | None = None,
        temporal: bool = False,
    ):
        self.a_root = Path(a_dir)
        self.files_a = _list_images_recursive(a_dir)
        self.files_b = _list_images_recursive(b_dir)
        self.mask_root = Path(mask_dir) if mask_dir else None
        self.size = size
        self.train = train
        self.temporal = temporal

        # Pre-compute which indices have a genuine "next frame":
        # same parent folder AND adjacent in the sorted list.
        self.has_next = [
            i + 1 < len(self.files_a)
            and self.files_a[i].parent == self.files_a[i + 1].parent
            for i in range(len(self.files_a))
        ]

    def __len__(self) -> int:
        return max(len(self.files_a), len(self.files_b))

    # ------------------------------------------------------------------ #
    def _load_rgb(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _load_mask(self, a_path: Path, hw: tuple[int, int]) -> Image.Image:
        """Load the protection mask aligned to ``a_path``, or zeros."""
        if self.mask_root is None:
            return Image.new("L", hw, 0)
        rel = a_path.relative_to(self.a_root).with_suffix(".png")
        mask_path = self.mask_root / rel
        if not mask_path.is_file():
            return Image.new("L", hw, 0)
        with Image.open(mask_path) as img:
            return img.convert("L")

    def _augment_group(self, images: list[Image.Image], mask: Image.Image):
        """Resize + (shared) random crop/flip for one temporal group.

        Everything that moves pixels must move ALL members identically:
        A_t, A_{t+1} and the mask.  Color is untouched (no jitter): the GAN
        is supposed to learn the color statistics itself, and jittering the
        input would just add noise to that estimate.
        """
        size = self.size
        # Resize the short side to 1.12*size, keeping aspect ratio; masks use
        # NEAREST so they stay binary (bilinear would create gray edges).
        images = [TF.resize(im, int(size * 1.12), TF.InterpolationMode.BICUBIC) for im in images]
        mask = TF.resize(mask, int(size * 1.12), TF.InterpolationMode.NEAREST)

        if self.train:
            i, j, h, w = RandomCrop.get_params(images[0], (size, size))
            flip = random.random() < 0.5
        else:
            # Deterministic centre crop at eval time.
            w0, h0 = images[0].size
            i, j, h, w = (h0 - size) // 2, (w0 - size) // 2, size, size
            flip = False

        out = []
        for im in images:
            im = TF.crop(im, i, j, h, w)
            if flip:
                im = TF.hflip(im)
            out.append(TF.normalize(TF.to_tensor(im), [0.5] * 3, [0.5] * 3))
        mask = TF.crop(mask, i, j, h, w)
        if flip:
            mask = TF.hflip(mask)
        # to_tensor maps 255 -> 1.0; threshold at 0.5 to re-binarise.
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        return out, mask_t

    # ------------------------------------------------------------------ #
    def __getitem__(self, index: int) -> dict:
        a_idx = index % len(self.files_a)
        a_path = self.files_a[a_idx]

        use_pair = self.temporal and self.has_next[a_idx]
        a_img = self._load_rgb(a_path)
        mask_img = self._load_mask(a_path, a_img.size)
        next_img = self._load_rgb(self.files_a[a_idx + 1]) if use_pair else a_img

        (a_t, next_t), mask_t = self._augment_group([a_img, next_img], mask_img)

        # Domain B: independent random draw + independent augmentation.
        b_path = self.files_b[random.randint(0, len(self.files_b) - 1)] if self.train \
            else self.files_b[index % len(self.files_b)]
        (b_t,), _ = self._augment_group([self._load_rgb(b_path)], Image.new("L", (self.size, self.size), 0))

        return {
            "A": a_t,
            "B": b_t,
            "mask_a": mask_t,
            "A_next": next_t,
            "pair_valid": torch.tensor(use_pair, dtype=torch.bool),
        }
