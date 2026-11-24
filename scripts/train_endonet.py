#!/usr/bin/env python
"""Train EndoNet (tool detection + phase recognition) on Cholec80 frames.

Frame-level (default) or sequence-level (``--temporal lstm|gru``)::

    python scripts/train_endonet.py --backbone resnet50 --epochs 30 --batch-size 64
    python scripts/train_endonet.py --temporal lstm --seq-len 16 --seq-stride 4

Outputs ``runs/<name>/`` with ``metrics.csv``, ``args.json`` and
``checkpoints/{best,last}.pt``.
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

from cholec80_dl.data import (  # noqa: E402
    FrameDataset,
    SequenceDataset,
    build_transform,
    collate_frames,
    collate_sequences,
)
from cholec80_dl.engine import CheckpointManager, autocast, make_amp, resume_if_requested  # noqa: E402
from cholec80_dl.evaluation import evaluate_frame_model, scalar_metrics, split_outputs  # noqa: E402
from cholec80_dl.losses import EndoNetLoss, TemporalEndoNetLoss, tool_pos_weight  # noqa: E402
from cholec80_dl.metrics import PhaseMetrics  # noqa: E402
from cholec80_dl.models import EndoNet, TemporalEndoNet  # noqa: E402
from cholec80_dl.utils import (  # noqa: E402
    CSVLogger,
    build_optimizer,
    count_parameters,
    describe_device,
    dump_json,
    format_millions,
    make_run_dir,
    resolve_device,
    set_seed,
    setup_logging,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--frames-dir", default="data/frames")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--limit", type=int, default=0, help="use only N samples per split (debug)")
    # model
    p.add_argument("--backbone", default="resnet50", help="any torchvision model name")
    p.add_argument("--no-pretrained", action="store_true", help="random init (offline / ablation)")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--hidden", type=int, default=512, help="FC hidden width of both heads")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-tool-for-phase", action="store_true", help="phase head uses features only")
    # temporal
    p.add_argument("--temporal", choices=["none", "lstm", "gru"], default="none")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--seq-stride", type=int, default=4)
    p.add_argument("--rnn-hidden", type=int, default=256)
    p.add_argument("--rnn-layers", type=int, default=2)
    p.add_argument("--unidirectional", action="store_true")
    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--freeze-epochs", type=int, default=1, help="linear-probe warm-up epochs")
    p.add_argument("--label-smoothing", type=float, default=0.0)
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
    p.add_argument("--out", default="runs", help="run root")
    p.add_argument("--name", default="endonet")
    p.add_argument("--monitor", default="val_phase_f1",
                   choices=["val_phase_f1", "val_phase_acc", "val_tool_map"])
    p.add_argument("--patience", type=int, default=8, help="early-stop patience (0 = off)")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--tag", default=None, help="suffix appended to the run name")
    return p.parse_args(argv)


def resolve_csv(args, split: str) -> Path:
    override = getattr(args, f"{split}_csv")
    return Path(override) if override else Path(args.index_dir) / f"{split}.csv"


def build_model(args) -> torch.nn.Module:
    frame = EndoNet(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        input_size=args.size,
        hidden=args.hidden,
        dropout=args.dropout,
        use_tool_for_phase=not args.no_tool_for_phase,
    )
    if args.temporal == "none":
        return frame
    return TemporalEndoNet(
        frame,
        hidden=args.rnn_hidden,
        num_layers=args.rnn_layers,
        cell=args.temporal,
        bidirectional=not args.unidirectional,
        dropout=args.dropout,
    )


def model_config(args) -> dict:
    return {
        "backbone": args.backbone,
        "pretrained": not args.no_pretrained,
        "input_size": args.size,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "use_tool_for_phase": not args.no_tool_for_phase,
        "arch": "endonet",
        "temporal": args.temporal,
        "rnn_hidden": args.rnn_hidden,
        "rnn_layers": args.rnn_layers,
        "bidirectional": not args.unidirectional,
    }


def build_loader(dataset, batch_size, train, args, collate):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=collate,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=None if args.workers == 0 else _worker_init(args.seed),
        generator=generator,
    )


def _worker_init(seed: int):
    def fn(_):
        import random

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    return fn


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    print(f"device: {describe_device(device)} | amp: {amp}")

    temporal = args.temporal != "none"
    limit = args.limit or None

    train_df = pd.read_csv(resolve_csv(args, "train"))
    val_df = pd.read_csv(resolve_csv(args, "val")) if resolve_csv(args, "val").exists() else None
    test_csv = resolve_csv(args, "test")

    ds_cls = SequenceDataset if temporal else FrameDataset
    common = dict(frames_root=args.frames_dir, size=args.size)
    if temporal:
        common.update(seq_len=args.seq_len, seq_stride=args.seq_stride)
        collate = collate_sequences
    else:
        collate = collate_frames

    train_ds = ds_cls(train_df, train=True, augment=args.augment, limit=limit, **common)
    val_ds = ds_cls(val_df, train=False, limit=limit, **common) if val_df is not None and len(val_df) else None
    print(f"train samples: {len(train_ds)}" + (f" | val samples: {len(val_ds)}" if val_ds else ""))

    train_loader = build_loader(train_ds, args.batch_size, True, args, collate)
    val_loader = build_loader(val_ds, args.batch_size, False, args, collate) if val_ds else None

    model = build_model(args).to(device)
    print(f"model: {type(model).__name__} ({format_millions(count_parameters(model))} trainable params)")

    optimizer = build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
        backbone_lr_mult=args.backbone_lr_mult, backbone_prefix="backbone",
    )
    _, scaler = make_amp(device, amp)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _warmup_cosine(args.epochs, len(train_loader), warmup_epochs=1.0)
    )

    # class balancing computed from the training split only
    phase_weights = None
    if not args.no_phase_balance:
        counts = np.bincount(train_df["phase"].to_numpy(), minlength=7).astype(np.float64)
        counts[counts == 0] = counts[counts > 0].min()
        phase_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    pos_weight = None if args.no_tool_balance else tool_pos_weight(train_df)

    loss_cls = TemporalEndoNetLoss if temporal else EndoNetLoss
    criterion = loss_cls(
        phase_weight=args.phase_weight,
        tool_weight=args.tool_weight,
        phase_class_weights=phase_weights,
        tool_pos_weight=pos_weight,
        label_smoothing=args.label_smoothing,
    ).to(device)

    name = args.name + (f"-{args.tag}" if args.tag else "")
    run_dir = make_run_dir(args.out, name, seed=args.seed)
    dump_json({"args": vars(args), "model": model_config(args), "device": str(device)}, run_dir / "args.json")
    logger = CSVLogger(
        run_dir / "metrics.csv",
        ["epoch", "lr", "train_loss", "train_phase_acc", "val_loss", "val_phase_acc",
         "val_phase_f1", "val_tool_map", "epoch_seconds"],
    )
    ckpt = CheckpointManager(
        run_dir / "checkpoints", monitor=args.monitor, mode="max",
        extra={"model_config": model_config(args), "args": vars(args)},
    )

    start_epoch = resume_if_requested(args.resume, model, optimizer, scaler, device)
    if start_epoch:
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    best_epoch, stale = -1, 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        frozen = epoch < args.freeze_epochs
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, amp, args.grad_clip, temporal, frozen,
        )

        val_metrics = {}
        tuned = {}
        if val_loader is not None:
            result = evaluate_frame_model(model, val_loader, device, criterion, amp, temporal)
            val_metrics = scalar_metrics(result.metrics)
            tuned = {
                "val_phase_f1": val_metrics.get("phase_f1_macro", float("nan")),
                "val_phase_acc": val_metrics.get("phase_acc", float("nan")),
                "val_tool_map": val_metrics.get("tool_map", float("nan")),
                "val_loss": val_metrics.get("loss", float("inf")),
            }
        else:
            tuned = {"val_phase_f1": train_stats["phase_acc"], "val_phase_acc": train_stats["phase_acc"],
                     "val_tool_map": float("nan"), "val_loss": train_stats["loss"]}
        improved = ckpt.step(model, tuned, epoch, optimizer, scaler)

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_phase_acc": train_stats["phase_acc"],
            "val_loss": val_metrics.get("loss"),
            "val_phase_acc": val_metrics.get("phase_acc"),
            "val_phase_f1": val_metrics.get("phase_f1_macro"),
            "val_tool_map": val_metrics.get("tool_map"),
            "epoch_seconds": round(time.time() - t0, 1),
        }
        logger.log(row)

        val_acc = row["val_phase_acc"] if row["val_phase_acc"] is not None else float("nan")
        msg = f"[{epoch + 1:3d}/{args.epochs}] loss {row['train_loss']:.3f} val_acc {val_acc:.3f}"
        if row["val_phase_f1"] is not None:
            msg += f" val_f1 {row['val_phase_f1']:.3f} val_mAP {row['val_tool_map']:.3f}"
        msg += f" | {row['epoch_seconds']}s" + (" *best" if improved else "")
        print(msg)

        if improved:
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"early stopping: no {args.monitor} improvement for {stale} epochs")
                break

    # ---- final report on the held-out test videos -------------------------
    if test_csv.exists():
        from cholec80_dl.utils import load_checkpoint

        load_checkpoint(run_dir / "checkpoints" / "best.pt", model, map_location=device)
        test_ds = ds_cls(pd.read_csv(test_csv), train=False, limit=limit, **common)
        test_loader = build_loader(test_ds, args.batch_size, False, args, collate)
        result = evaluate_frame_model(model, test_loader, device, criterion, amp, temporal)
        test_metrics = scalar_metrics(result.metrics)
        dump_json({"test_metrics": test_metrics, "best_epoch": best_epoch + 1}, run_dir / "test_metrics.json")
        print(f"test phase_acc {test_metrics.get('phase_acc', float('nan')):.3f} "
              f"phase_f1 {test_metrics.get('phase_f1_macro', float('nan')):.3f} "
              f"tool_mAP {test_metrics.get('tool_map', float('nan')):.3f}")

    print(f"run dir: {run_dir}")
    return 0


def _warmup_cosine(total_epochs: int, steps_per_epoch: int, warmup_epochs: float = 1.0):
    import math

    total = max(1, total_epochs * steps_per_epoch)
    warmup = int(warmup_epochs * steps_per_epoch)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device,
                    amp, grad_clip, temporal, frozen) -> dict:
    model.train()
    model.set_backbone_trainable(not frozen)
    if frozen:
        model.backbone.eval()  # keep BN statistics of the pretrained encoder intact

    loss_sum, n_seen = 0.0, 0
    phase_m = PhaseMetrics()

    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        target_tools = batch["tools"].to(device, non_blocking=True)
        target_phase = (batch["phase_seq"] if temporal else batch["phase"]).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            out = model(x)
            loss, _ = criterion(out, target_tools, target_phase)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_sum += float(loss.detach()) * x.size(0)
        n_seen += x.size(0)

        phase_logits, _ = split_outputs(out, temporal)
        phase_m.update(phase_logits.argmax(-1).cpu().numpy(), batch["phase"].numpy())

    metrics = phase_m.compute()
    return {"loss": loss_sum / max(1, n_seen), "phase_acc": metrics["phase_acc"]}


if __name__ == "__main__":
    raise SystemExit(main())
