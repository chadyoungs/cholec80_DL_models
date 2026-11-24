"""Task losses (logit-level, AMP-safe, with optional class balancing)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from .constants import TOOLS


class EndoNetLoss(nn.Module):
    """``bce(tool) + phase_weight * ce(phase)`` for frame-level EndoNet."""

    def __init__(
        self,
        phase_weight: float = 1.0,
        tool_weight: float = 1.0,
        phase_class_weights: torch.Tensor | None = None,
        tool_pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.phase_weight = phase_weight
        self.tool_weight = tool_weight
        self.phase_loss = nn.CrossEntropyLoss(
            weight=phase_class_weights, label_smoothing=label_smoothing
        )
        self.tool_loss = nn.BCEWithLogitsLoss(pos_weight=tool_pos_weight)

    def forward(self, outputs, tools: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, Mapping[str, float]]:
        l_tool = self.tool_loss(outputs.tool_logits, tools.float())
        l_phase = self.phase_loss(outputs.phase_logits, phase.long())
        loss = self.tool_weight * l_tool + self.phase_weight * l_phase
        return loss, {"loss_tool": float(l_tool.detach()), "loss_phase": float(l_phase.detach())}


class TemporalEndoNetLoss(nn.Module):
    """Tool loss on the last frame + phase CE averaged over every timestep."""

    def __init__(
        self,
        phase_weight: float = 1.0,
        tool_weight: float = 1.0,
        phase_class_weights: torch.Tensor | None = None,
        tool_pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.phase_weight = phase_weight
        self.tool_weight = tool_weight
        self.phase_loss = nn.CrossEntropyLoss(
            weight=phase_class_weights, label_smoothing=label_smoothing, ignore_index=-1
        )
        self.tool_loss = nn.BCEWithLogitsLoss(pos_weight=tool_pos_weight)

    def forward(self, outputs, tools: torch.Tensor, phase_seq: torch.Tensor):
        phase_logits, tool_logits = outputs
        b, t, p = phase_logits.shape
        l_phase = self.phase_loss(phase_logits.reshape(b * t, p), phase_seq.reshape(b * t).long())
        l_tool = self.tool_loss(tool_logits, tools.float())
        loss = self.tool_weight * l_tool + self.phase_weight * l_phase
        return loss, {"loss_tool": float(l_tool.detach()), "loss_phase": float(l_phase.detach())}


class SlowFastLoss(nn.Module):
    """For clip-level SlowFast: CE(phase) + BCE(tool), same weighting options."""

    def __init__(
        self,
        phase_weight: float = 1.0,
        tool_weight: float = 1.0,
        phase_class_weights: torch.Tensor | None = None,
        tool_pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.phase_weight = phase_weight
        self.tool_weight = tool_weight
        self.phase_loss = nn.CrossEntropyLoss(weight=phase_class_weights)
        self.tool_loss = nn.BCEWithLogitsLoss(pos_weight=tool_pos_weight)

    def forward(self, outputs, tools: torch.Tensor, phase: torch.Tensor):
        l_phase = self.phase_loss(outputs.phase_logits, phase.long())
        l_tool = self.tool_loss(outputs.tool_logits, tools.float())
        loss = self.tool_weight * l_tool + self.phase_weight * l_phase
        return loss, {"loss_phase": float(l_phase.detach()), "loss_tool": float(l_tool.detach())}


def tool_pos_weight(index, num_tools: int = 7) -> torch.Tensor:
    """BCE ``pos_weight = #negatives / #positives`` per instrument class."""
    values = index[list(TOOLS)].to_numpy(dtype=np.float64)
    positives = values.sum(axis=0)
    negatives = len(values) - positives
    weights = np.divide(negatives, positives, out=np.ones_like(positives), where=positives > 0)
    return torch.tensor(weights, dtype=torch.float32)
