"""SlowFast network for clip-level Cholec80 recognition.

Reference: Feichtenhofer et al., "SlowFast Networks for Video Recognition",
ICCV 2019.

The implementation follows the paper's channel schedule:

* slow pathway: ``slow_channels`` base width, sparse clip (``T_slow`` frames),
* fast pathway: ``slow_channels // beta_inv`` base width, dense clip (``alpha *
  T_slow`` frames),
* lateral connections after the stems and after every residual stage,
* dual heads: 7-way phase softmax + 7-way multi-label tool sigmoid.

Training from scratch on 80 videos is hopeless, so weights are optionally
initialised by inflating the 2-D ImageNet ResNet-50 kernels (slow **and** fast
pathway, channel-sliced) -- see :func:`inflate_from_imagenet`.
"""

from __future__ import annotations

import copy
from typing import NamedTuple

import torch
import torch.nn as nn
from torchvision import models as tvmodels

from ..constants import NUM_PHASES, NUM_TOOLS


class SlowFastOutput(NamedTuple):
    phase_logits: torch.Tensor  # (B, num_phases)
    tool_logits: torch.Tensor   # (B, num_tools)
    features: torch.Tensor      # (B, feat_dim)


class Bottleneck3D(nn.Module):
    """``1x1x1 -> 1x3x3 -> 1x1x1`` bottleneck with a temporal kernel in conv1."""

    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1,
                 t_kernel: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv3d(
            in_channels, mid_channels, kernel_size=(t_kernel, 1, 1),
            padding=(t_kernel // 2, 0, 0), bias=False,
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(
            mid_channels, mid_channels, kernel_size=(1, 3, 3),
            stride=(1, stride, stride), padding=(0, 1, 1), bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def make_stage(in_channels: int, mid_channels: int, num_blocks: int,
               stride: int = 1, t_kernel: int = 1) -> nn.Sequential:
    out_channels = mid_channels * Bottleneck3D.expansion
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1,
                      stride=(1, stride, stride), bias=False),
            nn.BatchNorm3d(out_channels),
        )
    blocks = [Bottleneck3D(in_channels, mid_channels, stride, t_kernel, downsample)]
    blocks += [Bottleneck3D(out_channels, mid_channels, 1, t_kernel) for _ in range(1, num_blocks)]
    return nn.Sequential(*blocks)


class LateralConnection(nn.Module):
    """Fast -> Slow fusion via a time-strided 3-D conv (``T -> C``)."""

    def __init__(self, fast_channels: int, alpha: int):
        super().__init__()
        self.conv = nn.Conv3d(
            fast_channels, fast_channels * 2, kernel_size=(5, 1, 1),
            stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False,
        )
        self.bn = nn.BatchNorm3d(fast_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        lateral = self.relu(self.bn(self.conv(fast)))
        # 1-frame slow clips / short T can make the strided conv one step shorter
        if lateral.shape[2] != slow.shape[2]:
            n = min(lateral.shape[2], slow.shape[2])
            lateral, slow = lateral[:, :, :n], slow[:, :, :n]
        return torch.cat([slow, lateral], dim=1)


class SlowFast(nn.Module):
    """SlowFast with dual heads for Cholec80 (phase + tool)."""

    def __init__(
        self,
        num_phases: int = NUM_PHASES,
        num_tools: int = NUM_TOOLS,
        slow_channels: int = 64,
        beta_inv: int = 8,
        alpha: int = 4,
        dropout: float = 0.5,
        input_size: int = 224,
    ):
        super().__init__()
        self.alpha = alpha
        s = slow_channels
        f = slow_channels // beta_inv

        # stems
        self.slow_stem = nn.Sequential(
            nn.Conv3d(3, s, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(s), nn.ReLU(inplace=True),
        )
        self.fast_stem = nn.Sequential(
            nn.Conv3d(3, f, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(f), nn.ReLU(inplace=True),
        )
        self.slow_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # lateral + residual stages (channel arithmetic mirrors the paper)
        self.lateral_0 = LateralConnection(f, alpha)
        self.slow_res2 = make_stage(s + 2 * f, s, 3, t_kernel=1)
        self.fast_res2 = make_stage(f, f, 3, t_kernel=3)

        self.lateral_1 = LateralConnection(4 * f, alpha)
        self.slow_res3 = make_stage(4 * s + 8 * f, 2 * s, 4, stride=2, t_kernel=1)
        self.fast_res3 = make_stage(4 * f, 2 * f, 4, stride=2, t_kernel=3)

        self.lateral_2 = LateralConnection(8 * f, alpha)
        self.slow_res4 = make_stage(8 * s + 16 * f, 4 * s, 6, stride=2, t_kernel=3)
        self.fast_res4 = make_stage(8 * f, 4 * f, 6, stride=2, t_kernel=3)

        self.lateral_3 = LateralConnection(16 * f, alpha)
        self.slow_res5 = make_stage(16 * s + 32 * f, 8 * s, 3, stride=2, t_kernel=3)
        self.fast_res5 = make_stage(16 * f, 8 * f, 3, stride=2, t_kernel=3)

        self.feat_dim = 32 * s + 32 * f
        self.dropout = nn.Dropout(dropout)
        self.phase_head = nn.Linear(self.feat_dim, num_phases)
        self.tool_head = nn.Linear(self.feat_dim, num_tools)
        self.input_size = input_size
        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, slow_x: torch.Tensor, fast_x: torch.Tensor) -> SlowFastOutput:
        s = self.slow_pool(self.slow_stem(slow_x))
        f = self.fast_pool(self.fast_stem(fast_x))

        s = self.slow_res2(self.lateral_0(f, s))
        f = self.fast_res2(f)

        s = self.slow_res3(self.lateral_1(f, s))
        f = self.fast_res3(f)

        s = self.slow_res4(self.lateral_2(f, s))
        f = self.fast_res4(f)

        s = self.slow_res5(self.lateral_3(f, s))
        f = self.fast_res5(f)

        s = s.mean(dim=(2, 3, 4))
        f = f.mean(dim=(2, 3, 4))
        feat = self.dropout(torch.cat([s, f], dim=1))
        return SlowFastOutput(self.phase_head(feat), self.tool_head(feat), feat)

    # ------------------------------------------------------------------ #
    def inflate_from_imagenet(self, resnet_name: str = "resnet50", verbose: bool = True) -> None:
        """Copy 2-D ImageNet ResNet weights into both 3-D pathways.

        Temporal kernels are inflated by repeating the 2-D kernel along time and
        dividing by the kernel length (so a constant input keeps its response).
        The fast pathway keeps the *first* ``f_channel`` filters (channel slicing).
        """
        ref = tvmodels.get_model(resnet_name, weights="DEFAULT").eval()
        stages = [
            (self.slow_res2, self.fast_res2, ref.layer1),
            (self.slow_res3, self.fast_res3, ref.layer2),
            (self.slow_res4, self.fast_res4, ref.layer3),
            (self.slow_res5, self.fast_res5, ref.layer4),
        ]
        _inflate_conv(self.slow_stem[0], ref.conv1.weight, self.slow_stem[1], ref.bn1)
        _inflate_conv(self.fast_stem[0], ref.conv1.weight, self.fast_stem[1], ref.bn1)
        for slow_stage, fast_stage, ref_stage in stages:
            _inflate_stage(slow_stage, ref_stage)
            _inflate_stage(fast_stage, ref_stage)
        if verbose:
            print(f"[SlowFast] inflated 3-D weights from ImageNet {resnet_name} (slow+fast)")

    def set_backbone_trainable(self, flag: bool) -> None:
        """Freeze/unfreeze everything except the two heads."""
        for p in self.parameters():
            p.requires_grad = flag
        for p in list(self.phase_head.parameters()) + list(self.tool_head.parameters()):
            p.requires_grad = True


# --------------------------------------------------------------------------- #
# 2D -> 3D weight inflation helpers
# --------------------------------------------------------------------------- #

def _inflate_conv(conv3d: nn.Conv3d, w2d: torch.Tensor, bn3d: nn.BatchNorm3d, bn2d: nn.Module) -> None:
    out_c, in_c = conv3d.out_channels, conv3d.in_channels
    t = conv3d.kernel_size[0]
    w = w2d[:out_c, :in_c]
    w = w.unsqueeze(2).repeat(1, 1, t, 1, 1) / t
    with torch.no_grad():
        conv3d.weight.copy_(w)
        for attr in ("weight", "bias"):
            src = getattr(bn2d, attr, None)
            if src is not None:
                getattr(bn3d, attr).copy_(src[: conv3d.out_channels])
        for attr in ("running_mean", "running_var"):
            src = getattr(bn2d, attr, None)
            if src is not None:
                getattr(bn3d, attr).copy_(src[: conv3d.out_channels])


def _inflate_bottleneck(b3d: Bottleneck3D, b2d: nn.Module) -> None:
    _inflate_conv(b3d.conv1, b2d.conv1.weight, b3d.bn1, b2d.bn1)
    _inflate_conv(b3d.conv2, b2d.conv2.weight, b3d.bn2, b2d.bn2)
    _inflate_conv(b3d.conv3, b2d.conv3.weight, b3d.bn3, b2d.bn3)
    if b3d.downsample is not None:
        _inflate_conv(b3d.downsample[0], b2d.downsample[0].weight,
                      b3d.downsample[1], b2d.downsample[1])


def _inflate_stage(stage3d: nn.Sequential, stage2d: nn.Module) -> None:
    for b3d, b2d in zip(stage3d, stage2d):
        _inflate_bottleneck(b3d, b2d)


def build_slowfast(**kwargs) -> SlowFast:
    return SlowFast(**kwargs)
