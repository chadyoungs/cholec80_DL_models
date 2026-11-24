"""Checkpoint round-trip, CSV logging, optimizer grouping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from cholec80_dl.engine import CheckpointManager
from cholec80_dl.utils import CSVLogger, build_optimizer, count_parameters, load_checkpoint, save_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))


def test_checkpoint_roundtrip(tmp_path):
    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(tmp_path / "ckpt.pt", model, optimizer=opt, epoch=3, best=0.8,
                    extra={"model_config": {"backbone": "tiny"}})

    fresh = TinyModel()
    ckpt = load_checkpoint(tmp_path / "ckpt.pt", fresh, optimizer=torch.optim.AdamW(fresh.parameters()))
    assert ckpt["epoch"] == 3 and ckpt["best"] == 0.8
    assert ckpt["model_config"]["backbone"] == "tiny"
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        assert torch.allclose(a, b)


def test_checkpoint_manager_keeps_best(tmp_path):
    model = TinyModel()
    manager = CheckpointManager(tmp_path, monitor="val_f1", mode="max")
    manager.step(model, {"val_f1": 0.5}, epoch=0)
    manager.step(model, {"val_f1": 0.4}, epoch=1)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    assert load_checkpoint(tmp_path / "best.pt")["epoch"] == 0
    manager.step(model, {"val_f1": 0.9}, epoch=2)
    assert load_checkpoint(tmp_path / "best.pt")["epoch"] == 2


def test_csv_logger_single_header(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = CSVLogger(path, ["epoch", "loss"])
    logger.log({"epoch": 1, "loss": 0.5, "ignored": 1})
    logger.log({"epoch": 2, "loss": 0.25})
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "epoch,loss"
    assert len(lines) == 3
    assert lines[1] == "1,0.500000"


def test_build_optimizer_groups_by_prefix():
    model = TinyModel()
    opt = build_optimizer(model, lr=1e-3, backbone_lr_mult=0.1)
    groups = {tuple(sorted(g["params"][0].shape)): g["lr"] for g in opt.param_groups}
    assert any(abs(lr - 1e-4) < 1e-9 for lr in groups.values())
    assert any(abs(lr - 1e-3) < 1e-9 for lr in groups.values())
    assert count_parameters(model) == 4 * 4 + 4 + 4 * 2 + 2
