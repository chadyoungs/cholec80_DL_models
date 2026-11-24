"""EndoNet: shared CNN encoder + tool head + phase head.

Differences from the legacy implementation:

* heads emit **logits** (no ``sigmoid`` inside the model), so
  ``BCEWithLogitsLoss`` / ``CrossEntropyLoss`` and AMP behave correctly;
* the encoder is any torchvision backbone (``resnet50`` default) instead of a
  hard-coded AlexNet;
* the phase head consumes ``[features, sigmoid(tool)]`` exactly like the
  original paper, but the split is configurable and the dimension is derived
  from the backbone instead of a magic 4103;
* an optional temporal wrapper (LSTM / BiLSTM) turns the frame model into a
  sequence model (HRNN-style phase recognition).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from ..constants import NUM_PHASES, NUM_TOOLS
from .backbone import build_backbone


class EndoNetOutput(NamedTuple):
    features: torch.Tensor      # (B, D)
    tool_logits: torch.Tensor   # (B, num_tools)
    phase_logits: torch.Tensor  # (B, num_phases)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EndoNet(nn.Module):
    """Frame-level EndoNet (tool detection + phase classification)."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        input_size: int = 224,
        num_phases: int = NUM_PHASES,
        num_tools: int = NUM_TOOLS,
        hidden: int = 512,
        dropout: float = 0.3,
        use_tool_for_phase: bool = True,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained, input_size=input_size)
        feat_dim = self.backbone.out_dim
        self.use_tool_for_phase = use_tool_for_phase
        self.num_tools = num_tools

        self.tool_head = _MLP(feat_dim, hidden, num_tools, dropout)
        phase_in = feat_dim + (num_tools if use_tool_for_phase else 0)
        self.phase_head = _MLP(phase_in, hidden, num_phases, dropout)

    # ------------------------------------------------------------------ #
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def set_backbone_trainable(self, flag: bool) -> None:
        self.backbone.set_trainable(flag)

    def forward(self, x: torch.Tensor) -> EndoNetOutput:
        feats = self.backbone(x)
        tool_logits = self.tool_head(feats)
        phase_in = (
            torch.cat([feats, torch.sigmoid(tool_logits)], dim=1)
            if self.use_tool_for_phase
            else feats
        )
        phase_logits = self.phase_head(phase_in)
        return EndoNetOutput(feats, tool_logits, phase_logits)


class TemporalEndoNet(nn.Module):
    """EndoNet + recurrent head over frame features (phase per timestep).

    ``forward`` returns ``(phase_logits[B,T,P], tool_logits[B,tools])`` where the
    tool prediction is taken at the last timestep.
    """

    def __init__(
        self,
        frame_model: EndoNet,
        hidden: int = 256,
        num_layers: int = 2,
        cell: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        if cell not in {"lstm", "gru"}:
            raise ValueError(f"cell must be lstm|gru, got {cell!r}")
        self.frame_model = frame_model
        self.num_phases = frame_model.phase_head.net[-1].out_features

        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=frame_model.backbone.out_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.phase_head = _MLP(hidden * (2 if bidirectional else 1), hidden, self.num_phases, dropout)

    @property
    def backbone(self):  # noqa: N802 - convenience for optimisation helpers
        return self.frame_model.backbone

    def set_backbone_trainable(self, flag: bool) -> None:
        self.frame_model.set_backbone_trainable(flag)

    def forward(self, x: torch.Tensor):
        b, t = x.shape[:2]
        feats = self.frame_model.backbone(x.flatten(0, 1))          # (B*T, D)
        feats = feats.view(b, t, -1)
        seq, _ = self.rnn(feats)                                    # (B,T,H)
        phase_logits = self.phase_head(seq.reshape(b * t, -1)).view(b, t, -1)   # (B,T,P)
        tool_logits = self.frame_model.tool_head(feats[:, -1])       # (B,tools)
        return phase_logits, tool_logits
