"""CycleGAN (ResNet generator / PatchGAN discriminator), modernised.

Based on Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks" (ICCV 2017), following the reference implementation used
by the previous ``CycleGAN/`` folder -- rewritten without ``Variable`` or
Visdom and with proper ``nn.Module`` initialisation helpers.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """ResNet generator (9 blocks by default, ``ngf=64``)."""

    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        ch = ngf
        for _ in range(2):  # downsampling
            layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2
        layers += [ResidualBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):  # upsampling
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field for 256px inputs)."""

    def __init__(self, input_nc: int = 3, ndf: int = 64):
        super().__init__()

        def block(in_c: int, out_c: int, normalize: bool = True) -> list[nn.Module]:
            ops: list[nn.Module] = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if normalize:
                ops.append(nn.InstanceNorm2d(out_c))
            ops.append(nn.LeakyReLU(0.2, inplace=True))
            return ops

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *block(ndf, ndf * 2),
            *block(ndf * 2, ndf * 4),
            nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, padding=1),   # patch output, no sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).flatten(1)


class ReplayBuffer:
    """History of generated images used to stabilise discriminator training."""

    def __init__(self, max_size: int = 50):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.data: list[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        selected: list[torch.Tensor] = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(image)
                selected.append(image)
            elif random.random() > 0.5:
                i = random.randrange(self.max_size)
                selected.append(self.data[i].clone())
                self.data[i] = image
            else:
                selected.append(image)
        return torch.cat(selected, dim=0)


def weights_init_normal(module: nn.Module) -> None:
    """DCGAN-style init; skips parameter-free layers (``affine=False`` norms)."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1 and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("InstanceNorm") != -1:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


def linear_decay_lambda(n_epochs: int, decay_start_epoch: int):
    """LR multiplier: constant until ``decay_start_epoch``, then linear to 0."""
    if decay_start_epoch >= n_epochs:
        raise ValueError("decay_start_epoch must be < n_epochs")

    def fn(epoch: int) -> float:
        return 1.0 - max(0, epoch - decay_start_epoch) / (n_epochs - decay_start_epoch)

    return fn
