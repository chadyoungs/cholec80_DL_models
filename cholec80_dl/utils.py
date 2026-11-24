"""Shared utilities: seeding, devices, logging, metrics plumbing, checkpoints."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

logger = logging.getLogger("cholec80")


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def supports_amp(device: torch.device) -> bool:
    return device.type == "cuda"


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return f"cuda:{device.index or 0} ({name}, {mem:.1f} GB)"
    return "cpu"


# --------------------------------------------------------------------------- #
# Logging / output dirs
# --------------------------------------------------------------------------- #

def setup_logging(level: int = logging.INFO) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)


def make_run_dir(root: str | Path, name: str, seed: int = 42, timestamp: bool = True) -> Path:
    """Create ``root/<name>[_YYYYmmdd-HHMMSS]`` and return it."""
    root = Path(root)
    run_name = name
    if timestamp:
        run_name = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def dump_json(obj: Mapping[str, Any], path: str | Path) -> None:
    def _default(o: Any):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    Path(path).write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


class CSVLogger:
    """Append-only CSV logger with a fixed header (one row per epoch)."""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._header_written = self.path.exists()

    def log(self, row: Mapping[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (np.floating,)):
        return f"{float(value):.6f}"
    return value


class AverageMeter:
    """Running mean (last / avg / count)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.avg:.4f}"


# --------------------------------------------------------------------------- #
# Optimizer / checkpoints
# --------------------------------------------------------------------------- #

def build_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    backbone_lr_mult: float = 1.0,
    backbone_prefix: str = "backbone",
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """AdamW with an optional lower LR for the pretrained backbone."""
    backbone, head = [], []
    for name, param in model.named_parameters():
        (backbone if name.startswith(backbone_prefix) else head).append(param)
    groups = [{"params": head, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": lr * backbone_lr_mult})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=betas)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    best: float = float("-inf"),
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": _unwrap(model).state_dict(),
        "epoch": epoch,
        "best": best,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, (torch.nn.DataParallel,)) else model


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        _unwrap(model).load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_millions(n: int) -> str:
    return f"{n / 1e6:.1f}M"


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #

def seed_worker(worker_id: int) -> None:  # pragma: no cover - used by DataLoader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
