#!/usr/bin/env python
"""Train a SlowFast clip model on Cholec80 (phase + tool heads).

Clips are built from the *sampled-frame* index (1 fps by default), so
``--clip-len 64`` covers ~1 minute and ``--tau 8`` makes the slow view sample
every 8 sec::

    python scripts/train_slowfast.py --clip-len 64 --tau 8 --alpha 4 --epochs 20

By default the 3-D kernels are inflated from ImageNet ResNet-50 (both
pathways), which is the only practical way to train this architecture on
80 videos. Pass ``--no-inflate`` to train from scratch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cholec80.data import ClipDataset, collate_clips  # noqa: E402
from cholec80.engine import CheckpointManager, autocast, make_amp, resume_if_requested  # noqa: E402
from cholec80.evaluation import evaluate_frame_model, scalar_metrics  # noqa: E402
from cholec80.losses import SlowFastLoss, tool_pos_weight  # noqa: E402
from cholec80.metrics import PhaseMetrics  # noqa: E402
from cholec80.models import SlowFast  # noqa: E402
from cholec80.utils import (  # noqa: E402
    CSVLogger,
    build_optimizer,
    count_parameters,
    describe_device,
    dump_json,
    format_millions,
    make_run_dir,
    resolve_device,
    set_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--frames-dir", default="data/frames")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--limit", type=int, default=0)
    # clip sampling
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--clip-len", type=int, default=64, help="sampled frames per clip")
    p.add_argument("--tau", type=int, default=8, help="slow-pathway stride (in sampled frames)")
    p.add_argument("--alpha", type=int, default=4, help="T_fast / T_slow ratio")
    p.add_argument("--clip-stride", type=int, default=32)
    # model
    p.add_argument("--slow-channels", type=int, default=64)
    p.add_argument("--beta-inv", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no-inflate", action="store_true", help="do not load ImageNet ResNet-50 weights")
    p.add_argument("--inflate-from", default="resnet50")
    # optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--freeze-epochs", type=int, default=0)
    p.add_argument("--phase-weight", type=float, default=1.0)
    p.add_argument("--tool-weight", type=float, default=1.0)
    p.add_argument("--no-phase-balance", action="store_true")
    p.add_argument("--no-tool-balance", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    # bookkeeping
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default="slowfast")
    p.add_argument("--monitor", default="val_phase_f1",
                   choices=["val_phase_f1", "val_phase_acc", "val_tool_map"])
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--tag", default=None)
    return p.parse_args(argv)


def build_model(args) -> SlowFast:
    return SlowFast(
        slow_channels=args.slow_channels,
        beta_inv=args.beta_inv,
        alpha=args.alpha,
        dropout=args.dropout,
        input_size=args.size,
    )


def model_config(args) -> dict:
    return {
        "arch": "slowfast",
        "slow_channels": args.slow_channels,
        "beta_inv": args.beta_inv,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "input_size": args.size,
        "clip_len": args.clip_len,
        "tau": args.tau,
        "clip_stride": args.clip_stride,
    }


def make_loader(dataset, args, train: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
        pin_memory=True, drop_last=train, collate_fn=collate_clips,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        generator=generator,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    print(f"device: {describe_device(device)} | amp: {amp}")

    limit = args.limit or None
    clip_kwargs = dict(
        frames_root=args.frames_dir, size=args.size, clip_len=args.clip_len,
        tau=args.tau, alpha=args.alpha, clip_stride=args.clip_stride,
    )
    train_csv = Path(args.train_csv) if args.train_csv else Path(args.index_dir) / "train.csv"
    val_csv = Path(args.val_csv) if args.val_csv else Path(args.index_dir) / "val.csv"
    test_csv = Path(args.test_csv) if args.test_csv else Path(args.index_dir) / "test.csv"

    train_df = pd.read_csv(train_csv)
    train_ds = ClipDataset(train_df, train=True, augment=args.augment, limit=limit, **clip_kwargs)
    val_ds = None
    if val_csv.exists():
        val_df = pd.read_csv(val_csv)
        if len(val_df):
            val_ds = ClipDataset(val_df, train=False, limit=limit, **clip_kwargs)
    print(f"train clips: {len(train_ds)}" + (f" | val clips: {len(val_ds)}" if val_ds else ""))

    train_loader = make_loader(train_ds, args, True)
    val_loader = make_loader(val_ds, args, False) if val_ds else None

    model = build_model(args)
    if not args.no_inflate:
        model.inflate_from_imagenet(args.inflate_from)
    model.to(device)
    print(f"model: SlowFast ({format_millions(count_parameters(model))} params, "
          f"{train_ds.n_slow}xslow + {train_ds.n_fast}xfast frames)")

    optimizer = build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
        backbone_lr_mult=args.backbone_lr_mult, backbone_prefix="slow_",
    )
    _, scaler = make_amp(device, amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))

    phase_weights = None
    if not args.no_phase_balance:
        counts = np.bincount(train_df["phase"].to_numpy(), minlength=7).astype(np.float64)
        counts[counts == 0] = counts[counts > 0].min()
        phase_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    criterion = SlowFastLoss(
        phase_weight=args.phase_weight, tool_weight=args.tool_weight,
        phase_class_weights=phase_weights,
        tool_pos_weight=None if args.no_tool_balance else tool_pos_weight(train_df),
    ).to(device)

    name = args.name + (f"-{args.tag}" if args.tag else "")
    run_dir = make_run_dir(args.out, name, seed=args.seed)
    dump_json({"args": vars(args), "model": model_config(args)}, run_dir / "args.json")
    logger = CSVLogger(
        run_dir / "metrics.csv",
        ["epoch", "train_loss", "train_phase_acc", "val_loss", "val_phase_acc",
         "val_phase_f1", "val_tool_map", "epoch_seconds"],
    )
    ckpt = CheckpointManager(
        run_dir / "checkpoints", monitor=args.monitor, mode="max",
        extra={"model_config": model_config(args), "args": vars(args)},
    )

    start_epoch = resume_if_requested(args.resume, model, optimizer, scaler, device)
    stale = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        frozen = epoch < args.freeze_epochs
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                      scheduler, device, amp, args.grad_clip, frozen)

        if val_loader is not None:
            result = evaluate_frame_model(model, val_loader, device, criterion, amp)
            vm = scalar_metrics(result.metrics)
        else:
            vm = {}
        tuned = {
            "val_phase_f1": vm.get("phase_f1_macro", float("nan")),
            "val_phase_acc": vm.get("phase_acc", float("nan")),
            "val_tool_map": vm.get("tool_map", float("nan")),
            "val_loss": vm.get("loss", float("inf")),
        }
        improved = ckpt.step(model, tuned, epoch, optimizer, scaler)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_phase_acc": train_stats["phase_acc"],
            "val_loss": vm.get("loss"),
            "val_phase_acc": vm.get("phase_acc"),
            "val_phase_f1": vm.get("phase_f1_macro"),
            "val_tool_map": vm.get("tool_map"),
            "epoch_seconds": round(time.time() - t0, 1),
        }
        logger.log(row)
        nan = float("nan")
        acc = row["val_phase_acc"] if row["val_phase_acc"] is not None else nan
        f1 = row["val_phase_f1"] if row["val_phase_f1"] is not None else nan
        mAP = row["val_tool_map"] if row["val_tool_map"] is not None else nan
        print(f"[{epoch + 1:3d}/{args.epochs}] loss {row['train_loss']:.3f} "
              f"val_acc {acc:.3f} val_f1 {f1:.3f} val_mAP {mAP:.3f} "
              f"| {row['epoch_seconds']}s" + (" *best" if improved else ""))

        if not improved:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"early stopping after {stale} epochs without improvement")
                break

    if test_csv.exists():
        from cholec80.utils import load_checkpoint

        load_checkpoint(run_dir / "checkpoints" / "best.pt", model, map_location=device)
        test_ds = ClipDataset(pd.read_csv(test_csv), train=False, limit=limit, **clip_kwargs)
        test_loader = make_loader(test_ds, args, False)
        result = evaluate_frame_model(model, test_loader, device, criterion, amp)
        test_metrics = scalar_metrics(result.metrics)
        dump_json({"test_metrics": test_metrics}, run_dir / "test_metrics.json")
        print(f"test phase_acc {test_metrics.get('phase_acc', float('nan')):.3f} "
              f"phase_f1 {test_metrics.get('phase_f1_macro', float('nan')):.3f} "
              f"tool_mAP {test_metrics.get('tool_map', float('nan')):.3f}")

    print(f"run dir: {run_dir}")
    return 0


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device,
                    amp, grad_clip, frozen) -> dict:
    model.train()
    if hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(not frozen)

    loss_sum, n_seen = 0.0, 0
    phase_m = PhaseMetrics()
    for batch in loader:
        slow, fast = (t.to(device, non_blocking=True) for t in batch["inputs"])
        target_tools = batch["tools"].to(device, non_blocking=True)
        target_phase = batch["phase"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            out = model(slow, fast)
            loss, _ = criterion(out, target_tools, target_phase)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_sum += float(loss.detach()) * slow.size(0)
        n_seen += slow.size(0)
        phase_m.update(out.phase_logits.argmax(-1).cpu().numpy(), batch["phase"].numpy())

    return {"loss": loss_sum / max(1, n_seen), "phase_acc": phase_m.compute()["phase_acc"]}


if __name__ == "__main__":
    raise SystemExit(main())
