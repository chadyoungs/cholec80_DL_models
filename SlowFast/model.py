"""
SlowFast network for surgical phase recognition and tool detection on Cholec80.

Architecture reference:
    "SlowFast Networks for Video Recognition"
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He
    ICCV 2019 — https://arxiv.org/abs/1812.03982

Design:
  • Slow pathway — ResNet-50 backbone, low frame-rate (T_slow=8), high channel
    capacity (64 base channels).  Uses temporal kernels of size 1 at early
    stages and 3 at deeper stages.
  • Fast pathway — ResNet-50 backbone, high frame-rate (T_fast=32), 1/8 channel
    capacity (8 base channels).  Uses temporal kernels of size 3 throughout to
    capture fine motion cues.
  • Lateral connections (Fast→Slow) — after the stem pool and after each of
    Res2, Res3, Res4.  A time-strided 3-D convolution reduces the Fast temporal
    dimension to match the Slow dimension; the result is concatenated along the
    channel axis before the corresponding Slow residual stage.
  • Dual head — a 7-class softmax head for surgical phase recognition and a
    7-way sigmoid head for multi-label tool detection.
"""

import torch
import torch.nn as nn

import config


# ── Building blocks ────────────────────────────────────────────────────────────

class Bottleneck3D(nn.Module):
    """3-D bottleneck residual block (1×1×1 → t×3×3 → 1×1×1)."""

    expansion = 4

    def __init__(self, in_channels, mid_channels,
                 stride=1, t_kernel=1, downsample=None):
        """
        Args:
            in_channels:  input channel count.
            mid_channels: bottleneck (middle) channel count; output = mid * 4.
            stride:       spatial stride applied in the 3×3 conv.
            t_kernel:     temporal kernel size for the first 1×t×1 conv.
            downsample:   optional shortcut projection module.
        """
        super().__init__()
        out_channels = mid_channels * self.expansion

        # 1st conv: temporal mixing (t×1×1)
        self.conv1 = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=(t_kernel, 1, 1),
            padding=(t_kernel // 2, 0, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)

        # 2nd conv: spatial mixing (1×3×3)
        self.conv2 = nn.Conv3d(
            mid_channels, mid_channels,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)

        # 3rd conv: channel projection (1×1×1)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_stage(in_channels, mid_channels, num_blocks, stride=1, t_kernel=1):
    """Stack *num_blocks* Bottleneck3D blocks into a residual stage.

    Args:
        in_channels:  channels entering the first block.
        mid_channels: bottleneck width; output channels = mid_channels * 4.
        num_blocks:   number of residual blocks.
        stride:       spatial stride for the first block's 3×3 conv.
        t_kernel:     temporal kernel size used in all blocks of this stage.
    """
    out_channels = mid_channels * Bottleneck3D.expansion

    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=1,
                      stride=(1, stride, stride),
                      bias=False),
            nn.BatchNorm3d(out_channels),
        )

    blocks = [
        Bottleneck3D(in_channels, mid_channels,
                     stride=stride, t_kernel=t_kernel, downsample=downsample)
    ]
    for _ in range(1, num_blocks):
        blocks.append(Bottleneck3D(out_channels, mid_channels, t_kernel=t_kernel))

    return nn.Sequential(*blocks)


class LateralConnection(nn.Module):
    """Fast-to-Slow lateral fusion module (TtoC — time-to-channel).

    Reduces the Fast temporal dimension to match the Slow temporal dimension
    via a time-strided 3-D convolution, then concatenates along the channel
    axis so that the Slow pathway receives both its own features and the
    motion cues from the Fast pathway.

    Output channel count of *fast_feat* after projection = 2 × fast_channels.
    The fused Slow tensor has channels = slow_channels + 2 × fast_channels.
    """

    def __init__(self, fast_channels, alpha):
        """
        Args:
            fast_channels: channel count of the Fast pathway feature map.
            alpha:         temporal stride = T_fast / T_slow.
        """
        super().__init__()
        self.conv = nn.Conv3d(
            fast_channels, fast_channels * 2,
            kernel_size=(5, 1, 1),
            stride=(alpha, 1, 1),
            padding=(2, 0, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(fast_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fast_feat, slow_feat):
        """
        Args:
            fast_feat: (B, C_fast, T_fast, H, W)
            slow_feat: (B, C_slow, T_slow, H, W)

        Returns:
            Tensor of shape (B, C_slow + 2*C_fast, T_slow, H, W)
        """
        lateral = self.relu(self.bn(self.conv(fast_feat)))
        return torch.cat([slow_feat, lateral], dim=1)


# ── Main model ─────────────────────────────────────────────────────────────────

class SlowFastNet(nn.Module):
    """SlowFast network for Cholec80 surgical video understanding.

    Inputs:
        slow_x: (B, 3, T_slow, H, W)  — sparsely-sampled frames
        fast_x: (B, 3, T_fast, H, W)  — densely-sampled frames

    Outputs:
        phase_logits: (B, num_phases)  — raw logits for phase classification
        tool_logits:  (B, num_tools)   — raw logits for multi-label tool detection
    """

    def __init__(
        self,
        num_phases=config.num_phases,
        num_tools=config.num_tools,
        alpha=config.alpha,
        beta_inv=config.beta_inv,
        dropout=config.dropout,
    ):
        super().__init__()
        self.alpha = alpha

        # Slow: 64 base channels; Fast: 64 // beta_inv = 8 base channels
        s = config.slow_base_channels       # 64
        f = config.fast_base_channels       # 8  (= s // beta_inv)

        # ── Slow stem ────────────────────────────────────────────────
        # Temporal kernel = 1: each frame is processed independently at first
        self.slow_stem = nn.Sequential(
            nn.Conv3d(3, s, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(s),
            nn.ReLU(inplace=True),
        )
        self.slow_pool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                      stride=(1, 2, 2), padding=(0, 1, 1))

        # ── Fast stem ────────────────────────────────────────────────
        # Temporal kernel = 5: captures short-term motion in dense frame stream
        self.fast_stem = nn.Sequential(
            nn.Conv3d(3, f, kernel_size=(5, 7, 7),
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(f),
            nn.ReLU(inplace=True),
        )
        self.fast_pool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                      stride=(1, 2, 2), padding=(0, 1, 1))

        # ── Lateral 0: after pool, before Res2 ───────────────────────
        # Fast(f=8) → lateral(2f=16); Slow input = s + 2f = 80
        self.lateral_0 = LateralConnection(f, alpha)

        # ── Res2 ─────────────────────────────────────────────────────
        # Slow: 80 → 256  (3 blocks, t_kernel=1, no spatial stride)
        self.slow_res2 = _make_stage(s + 2 * f, s, num_blocks=3, t_kernel=1)
        # Fast: f → 4f = 32  (3 blocks, t_kernel=3)
        self.fast_res2 = _make_stage(f, f, num_blocks=3, t_kernel=3)

        # ── Lateral 1: after Res2, before Res3 ───────────────────────
        # Fast(4f=32) → lateral(8f=64); Slow input = 256 + 64 = 320
        self.lateral_1 = LateralConnection(f * 4, alpha)

        # ── Res3 ─────────────────────────────────────────────────────
        # Slow: 320 → 512  (4 blocks, t_kernel=1, spatial stride=2)
        self.slow_res3 = _make_stage(s * 4 + f * 8, s * 2,
                                     num_blocks=4, stride=2, t_kernel=1)
        # Fast: 32 → 64  (4 blocks, t_kernel=3, spatial stride=2)
        self.fast_res3 = _make_stage(f * 4, f * 2,
                                     num_blocks=4, stride=2, t_kernel=3)

        # ── Lateral 2: after Res3, before Res4 ───────────────────────
        # Fast(8f=64) → lateral(16f=128); Slow input = 512 + 128 = 640
        self.lateral_2 = LateralConnection(f * 8, alpha)

        # ── Res4 ─────────────────────────────────────────────────────
        # Slow: 640 → 1024  (6 blocks, t_kernel=3, spatial stride=2)
        self.slow_res4 = _make_stage(s * 8 + f * 16, s * 4,
                                     num_blocks=6, stride=2, t_kernel=3)
        # Fast: 64 → 128  (6 blocks, t_kernel=3, spatial stride=2)
        self.fast_res4 = _make_stage(f * 8, f * 4,
                                     num_blocks=6, stride=2, t_kernel=3)

        # ── Lateral 3: after Res4, before Res5 ───────────────────────
        # Fast(16f=128) → lateral(32f=256); Slow input = 1024 + 256 = 1280
        self.lateral_3 = LateralConnection(f * 16, alpha)

        # ── Res5 ─────────────────────────────────────────────────────
        # Slow: 1280 → 2048  (3 blocks, t_kernel=3, spatial stride=2)
        self.slow_res5 = _make_stage(s * 16 + f * 32, s * 8,
                                     num_blocks=3, stride=2, t_kernel=3)
        # Fast: 128 → 256  (3 blocks, t_kernel=3, spatial stride=2)
        self.fast_res5 = _make_stage(f * 16, f * 8,
                                     num_blocks=3, stride=2, t_kernel=3)

        # ── Classification heads ──────────────────────────────────────
        # Concatenation of global-average-pooled Slow (2048) and Fast (256) features
        feat_dim = s * 32 + f * 32   # 2048 + 256 = 2304
        self.dropout = nn.Dropout(dropout)
        self.phase_head = nn.Linear(feat_dim, num_phases)
        self.tool_head = nn.Linear(feat_dim, num_tools)

        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, slow_x, fast_x):
        """
        Args:
            slow_x: (B, 3, T_slow, H, W) — sparsely-sampled clip
            fast_x: (B, 3, T_fast, H, W) — densely-sampled clip

        Returns:
            phase_logits: (B, num_phases)
            tool_logits:  (B, num_tools)
        """
        # ── Stems ─────────────────────────────────────────────────────
        s = self.slow_pool(self.slow_stem(slow_x))   # (B, 64,  T_s, H/4, W/4)
        f = self.fast_pool(self.fast_stem(fast_x))   # (B, 8,   T_f, H/4, W/4)

        # ── Lateral 0 + Res2 ──────────────────────────────────────────
        s = self.lateral_0(f, s)   # (B, 80,  T_s, H/4,  W/4)
        s = self.slow_res2(s)      # (B, 256, T_s, H/4,  W/4)
        f = self.fast_res2(f)      # (B, 32,  T_f, H/4,  W/4)

        # ── Lateral 1 + Res3 ──────────────────────────────────────────
        s = self.lateral_1(f, s)   # (B, 320, T_s, H/4,  W/4)
        s = self.slow_res3(s)      # (B, 512, T_s, H/8,  W/8)
        f = self.fast_res3(f)      # (B, 64,  T_f, H/8,  W/8)

        # ── Lateral 2 + Res4 ──────────────────────────────────────────
        s = self.lateral_2(f, s)   # (B, 640, T_s, H/8,  W/8)
        s = self.slow_res4(s)      # (B, 1024,T_s, H/16, W/16)
        f = self.fast_res4(f)      # (B, 128, T_f, H/16, W/16)

        # ── Lateral 3 + Res5 ──────────────────────────────────────────
        s = self.lateral_3(f, s)   # (B, 1280,T_s, H/16, W/16)
        s = self.slow_res5(s)      # (B, 2048,T_s, H/32, W/32)
        f = self.fast_res5(f)      # (B, 256, T_f, H/32, W/32)

        # ── Global average pooling ─────────────────────────────────────
        s = s.mean(dim=[2, 3, 4])  # (B, 2048)
        f = f.mean(dim=[2, 3, 4])  # (B, 256)

        feat = self.dropout(torch.cat([s, f], dim=1))  # (B, 2304)

        return self.phase_head(feat), self.tool_head(feat)


# ── Quick sanity check ─────────────────────────────────────────────────────────

def _test_forward():
    """Verify output shapes with a dummy forward pass."""
    import config
    model = SlowFastNet()
    model.eval()

    B = 2
    slow_x = torch.zeros(B, 3, config.T_slow, config.img_h, config.img_w)
    fast_x = torch.zeros(B, 3, config.T_fast, config.img_h, config.img_w)

    with torch.no_grad():
        phase_logits, tool_logits = model(slow_x, fast_x)

    print(f'phase_logits: {phase_logits.shape}')  # (2, 7)
    print(f'tool_logits:  {tool_logits.shape}')   # (2, 7)
    assert phase_logits.shape == (B, config.num_phases)
    assert tool_logits.shape == (B, config.num_tools)
    print('Forward pass OK.')


if __name__ == '__main__':
    _test_forward()
