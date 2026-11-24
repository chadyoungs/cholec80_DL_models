"""Training engine helpers shared by every task script.

Only the plumbing lives here: AMP context, one generic optimisation step, and
checkpoint/early-stopping policy. Task-specific loops stay in ``scripts/`` so
they remain readable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from .utils import build_optimizer, load_checkpoint, save_checkpoint


@dataclass
class TrainSettings:
    epochs: int = 30
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    warmup_epochs: float = 1.0
    scheduler: str = "cosine"  # "cosine" | "none"
    patience: int = 0           # 0 = no early stopping
    monitor: str = "val_phase_f1"
    mode: str = "max"


def make_amp(device: torch.device, enabled: bool):
    """Return ``(autocast_ctx_factory, grad_scaler)``.

    AMP is a no-op on CPU, so callers can use this unconditionally.
    """
    use = bool(enabled) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use)
    return use, scaler


def autocast(device: torch.device, enabled: bool):
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: TrainSettings,
    steps_per_epoch: int,
):
    """Cosine decay with linear warm-up, stepped per *iteration*."""
    if settings.scheduler == "none":
        return None
    total_steps = max(1, settings.epochs * steps_per_epoch)
    warmup_steps = int(settings.warmup_epochs * steps_per_epoch)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class CheckpointManager:
    """Keeps ``last.pt`` always and ``best.pt`` by monitored metric."""

    def __init__(self, ckpt_dir, monitor: str, mode: str = "max", extra: Mapping | None = None):
        from pathlib import Path

        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.extra = dict(extra or {})

    def _better(self, value: float) -> bool:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        return value > self.best if self.mode == "max" else value < self.best

    def step(self, model, metrics: Mapping[str, float], epoch: int, optimizer=None, scaler=None) -> bool:
        improved = self._better(metrics.get(self.monitor))
        if improved:
            self.best = float(metrics[self.monitor])
        save_checkpoint(
            self.dir / "last.pt", model, optimizer=optimizer, scaler=scaler, epoch=epoch,
            best=self.best, extra=self.extra,
        )
        if improved:
            save_checkpoint(
                self.dir / "best.pt", model, optimizer=optimizer, scaler=scaler, epoch=epoch,
                best=self.best, extra=self.extra,
            )
        return improved


def resume_if_requested(
    path: str | None,
    model: nn.Module,
    optimizer=None,
    scaler=None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Load a checkpoint in place and return the next epoch index."""
    if not path:
        return 0
    ckpt = load_checkpoint(path, model, optimizer, scaler, map_location=device)
    return int(ckpt.get("epoch", 0)) + 1


__all__ = [
    "TrainSettings",
    "make_amp",
    "autocast",
    "build_scheduler",
    "build_optimizer",
    "CheckpointManager",
    "resume_if_requested",
]
