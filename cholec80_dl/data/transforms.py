"""Image transforms used by every pipeline.

Frames extracted by ``scripts/prepare_dataset.py`` are already square and
centre-cropped, so loading is a plain resize; the training transform adds mild
photometric augmentation only (geometric augmentation is kept conservative
because instrument appearance is a strong cue for tool presence).
"""

from __future__ import annotations

from torchvision import transforms

from ..constants import IMAGENET_MEAN, IMAGENET_STD


def build_transform(size: int = 224, train: bool = False, augment: bool = False):
    ops: list = []
    if train and augment:
        ops += [
            transforms.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
        ]
    else:
        ops.append(transforms.Resize((size, size)))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)
