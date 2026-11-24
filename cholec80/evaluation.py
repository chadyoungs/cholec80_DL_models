"""Evaluation helpers shared by training and standalone evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .engine import autocast
from .metrics import PhaseMetrics, ToolMetrics
from .utils import AverageMeter


@dataclass
class EvalResult:
    metrics: dict[str, float] = field(default_factory=dict)
    confusion: np.ndarray | None = None


def _to_device(inputs, device: torch.device):
    """Frame models take one tensor, SlowFast takes ``(slow, fast)``."""
    if isinstance(inputs, (tuple, list)):
        return tuple(t.to(device, non_blocking=True) for t in inputs)
    return inputs.to(device, non_blocking=True)


def batch_size_of(batch) -> int:
    return int(batch["tools"].shape[0])


def split_outputs(out, temporal: bool):
    """Normalise ``EndoNetOutput`` / ``SlowFastOutput`` / ``(phase, tool)`` tuples."""
    if hasattr(out, "phase_logits"):
        phase_logits, tool_logits = out.phase_logits, out.tool_logits
    else:
        phase_logits, tool_logits = out
    if temporal and phase_logits.dim() == 3:
        phase_logits = phase_logits[:, -1]
    return phase_logits, tool_logits


@torch.no_grad()
def evaluate_frame_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    amp: bool = True,
    temporal: bool = False,
    max_batches: int | None = None,
) -> EvalResult:
    """Phase + tool metrics (plus optional loss) over ``loader``."""
    model.eval()
    phase_m = PhaseMetrics()
    tool_m = ToolMetrics()
    loss_m = AverageMeter()

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inputs = _to_device(batch["inputs"], device)
        with autocast(device, amp):
            out = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
        phase_logits, tool_logits = split_outputs(out, temporal)

        if criterion is not None:
            target_phase = batch["phase_seq"].to(device) if temporal else batch["phase"].to(device)
            loss, _ = criterion(out, batch["tools"].to(device), target_phase)
            loss_m.update(float(loss), batch_size_of(batch))

        phase_m.update(phase_logits.argmax(-1).cpu().numpy(), batch["phase"].numpy())
        tool_m.update(torch.sigmoid(tool_logits.float()).cpu().numpy(), batch["tools"].numpy())

    metrics: dict[str, float] = {}
    metrics.update(phase_m.compute())
    metrics.update(tool_m.compute())
    if criterion is not None:
        metrics["loss"] = loss_m.avg
    return EvalResult(metrics=metrics, confusion=phase_m.confusion())


def scalar_metrics(metrics: dict) -> dict[str, float]:
    """Drop non-scalar / nan entries so the result is JSON-safe."""
    out = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float, np.floating, np.integer)):
            continue
        value = float(v)
        if np.isnan(value):
            continue
        out[k] = value
    return out
