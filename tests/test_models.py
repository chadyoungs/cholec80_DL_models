"""Forward-pass and weight-inflation tests (CPU, offline, tiny inputs)."""

from __future__ import annotations

import torch
from torch import nn

from cholec80_dl.losses import EndoNetLoss, SlowFastLoss, TemporalEndoNetLoss
from cholec80_dl.models import EndoNet, SlowFast, TemporalEndoNet
from cholec80_dl.models.slowfast import _inflate_conv


def test_endonet_forward_and_logits():
    model = EndoNet(backbone="resnet18", pretrained=False, input_size=64, hidden=64, dropout=0.1).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    assert out.features.shape == (2, 512)
    assert out.tool_logits.shape == (2, 7)
    assert out.phase_logits.shape == (2, 7)
    # heads must expose raw logits (not probabilities)
    assert out.phase_logits.exp().sum(dim=1).min() > 1.5


def test_endonet_loss_backward():
    model = EndoNet(backbone="resnet18", pretrained=False, input_size=64, hidden=64)
    criterion = EndoNetLoss()
    out = model(torch.zeros(2, 3, 64, 64))
    loss, scalars = criterion(
        out,
        tools=torch.zeros(2, 7),
        phase=torch.tensor([0, 3]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert set(scalars) == {"loss_tool", "loss_phase"}


def test_endonet_freeze_helpers():
    model = EndoNet(backbone="resnet18", pretrained=False, input_size=64)
    model.set_backbone_trainable(False)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.tool_head.parameters())
    model.set_backbone_trainable(True)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_temporal_endonet_forward_and_loss():
    frame = EndoNet(backbone="resnet18", pretrained=False, input_size=48, hidden=32, dropout=0.1)
    model = TemporalEndoNet(frame, hidden=32, num_layers=1, cell="lstm").eval()
    with torch.no_grad():
        phase_logits, tool_logits = model(torch.zeros(2, 3, 3, 48, 48))
    assert phase_logits.shape == (2, 3, 7)
    assert tool_logits.shape == (2, 7)

    criterion = TemporalEndoNetLoss()
    loss, _ = criterion((phase_logits, tool_logits), torch.zeros(2, 7), torch.zeros(2, 3, dtype=torch.long))
    assert torch.isfinite(loss)


def test_slowfast_forward():
    model = SlowFast(slow_channels=8, beta_inv=2, alpha=2, input_size=32, dropout=0.1).eval()
    # slow: T=2, fast: T=4 (alpha=2)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 2, 32, 32), torch.zeros(1, 3, 4, 32, 32))
    assert out.phase_logits.shape == (1, 7)
    assert out.tool_logits.shape == (1, 7)
    assert out.features.shape[1] == model.feat_dim


def test_slowfast_loss():
    model = SlowFast(slow_channels=8, beta_inv=2, alpha=2, input_size=32)
    out = model(torch.zeros(2, 3, 2, 32, 32), torch.zeros(2, 3, 4, 32, 32))
    criterion = SlowFastLoss()
    loss, scalars = criterion(out, torch.zeros(2, 7), torch.tensor([1, 2]))
    assert torch.isfinite(loss) and set(scalars) == {"loss_phase", "loss_tool"}


def test_inflate_conv_from_2d_weights():
    conv3d = nn.Conv3d(4, 8, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
    bn3d = nn.BatchNorm3d(8)
    bn2d = nn.BatchNorm2d(8)
    w2d = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4, 1, 1)
    with torch.no_grad():
        bn2d.weight.fill_(2.0)
    _inflate_conv(conv3d, w2d, bn3d, bn2d)
    # kernel (t,1,1) sums over the temporal taps, so the sum must equal the 2-D kernel
    inflated_sum = conv3d.weight.sum(dim=2).squeeze(-1).squeeze(-1)
    assert torch.allclose(inflated_sum, w2d[:, :, 0, 0])
    assert torch.allclose(conv3d.weight[:, :, 0, 0, 0], w2d[:, :, 0, 0] / conv3d.kernel_size[0])
    assert torch.allclose(bn3d.weight, torch.full_like(bn3d.weight, 2.0))
