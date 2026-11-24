#!/usr/bin/env python
"""Pool-based active learning for EndoNet phase/tool recognition.

Each round: fine-tune on the currently labeled subset (warm start by default),
evaluate on the held-out *validation* videos, then query the most informative
unlabeled frames with one of seven strategies.

    python scripts/train_active.py --strategy entropy --n-rounds 8 --query-budget 200
    python scripts/train_active.py --strategy coreset --query-budget 100
    python scripts/train_active.py --strategy mc_dropout --mc-passes 10

Results: ``runs/al-<strategy>/curve.csv`` (one row per round) and ``.json``.
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

from cholec80.active_learning import ALPool, CandidateSet, STRATEGIES, query_indices  # noqa: E402
from cholec80.data import FrameDataset, collate_frames  # noqa: E402
from cholec80.engine import autocast, make_amp  # noqa: E402
from cholec80.evaluation import evaluate_frame_model, scalar_metrics  # noqa: E402
from cholec80.inference import mc_dropout_probs, score_loader  # noqa: E402
from cholec80.losses import EndoNetLoss, tool_pos_weight  # noqa: E402
from cholec80.metrics import PhaseMetrics  # noqa: E402
from cholec80.models import EndoNet  # noqa: E402
from cholec80.utils import (  # noqa: E402
    AverageMeter,
    CSVLogger,
    build_optimizer,
    describe_device,
    dump_json,
    make_run_dir,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--frames-dir", default="data/frames")
    p.add_argument("--train-csv", default=None, help="the unlabeled/labeled pool (defaults to train.csv)")
    p.add_argument("--val-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--limit", type=int, default=0)
    # active learning
    p.add_argument("--strategy", default="entropy", choices=sorted(STRATEGIES))
    p.add_argument("--n-rounds", type=int, default=8)
    p.add_argument("--query-budget", type=int, default=200, help="frames labeled per round")
    p.add_argument("--initial-ratio", type=float, default=0.05)
    p.add_argument("--epochs-per-round", type=int, default=5)
    p.add_argument("--fresh-model", action="store_true", help="re-init the model every round (cold start)")
    p.add_argument("--mc-passes", type=int, default=10, help="MC-dropout passes for BALD")
    p.add_argument("--badge-proj-dim", type=int, default=64)
    p.add_argument("--query-batch-size", type=int, default=64)
    # model / optimisation
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-balance", action="store_true")
    # bookkeeping
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs")
    p.add_argument("--tag", default=None)
    return p.parse_args(argv)


def model_config(args) -> dict:
    return {
        "arch": "endonet",
        "temporal": "none",
        "backbone": args.backbone,
        "pretrained": not args.no_pretrained,
        "input_size": args.size,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "use_tool_for_phase": True,
    }


def build_model(args) -> EndoNet:
    return EndoNet(
        backbone=args.backbone, pretrained=not args.no_pretrained, input_size=args.size,
        hidden=args.hidden, dropout=args.dropout, use_tool_for_phase=True,
    )


def build_loader(dataset, batch_size, train, workers, drop_last=False, collate=collate_frames):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=workers, pin_memory=True,
        drop_last=drop_last, collate_fn=collate, persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    print(f"device: {describe_device(device)} | strategy: {args.strategy} | amp: {amp}")

    def csv_path(split: str) -> Path:
        override = getattr(args, f"{split}_csv")
        return Path(override) if override else Path(args.index_dir) / f"{split}.csv"

    train_df = pd.read_csv(csv_path("train"))
    if args.limit:
        train_df = train_df.iloc[: args.limit]

    val_path = csv_path("val")
    if not val_path.exists() or not len(pd.read_csv(val_path)):
        val_path = csv_path("test")
        if not val_path.exists():
            raise SystemExit("need val.csv or test.csv for model selection during active learning")
        print(f"[warn] no validation split: using {val_path} for per-round evaluation")
    val_df = pd.read_csv(val_path)
    if args.limit:
        val_df = val_df.iloc[: args.limit]

    pool = ALPool(
        train_df, frames_root=args.frames_dir, initial_ratio=args.initial_ratio,
        seed=args.seed, size=args.size, augment=args.augment,
    )
    print(f"pool: {pool.n_total} frames | initial labeled: {pool.n_labeled}")

    val_loader = build_loader(
        FrameDataset(val_df, frames_root=args.frames_dir, size=args.size),
        args.batch_size, False, args.workers,
    )

    def fresh_components():
        model = build_model(args).to(device)
        optimizer = build_optimizer(
            model, lr=args.lr, weight_decay=args.weight_decay,
            backbone_lr_mult=args.backbone_lr_mult, backbone_prefix="backbone",
        )
        _, scaler = make_amp(device, amp)
        phase_weights = None
        if not args.no_balance:
            counts = np.bincount(train_df["phase"].to_numpy(), minlength=7).astype(np.float64)
            counts[counts == 0] = counts[counts > 0].min()
            phase_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        criterion = EndoNetLoss(
            phase_class_weights=phase_weights,
            tool_pos_weight=None if args.no_balance else tool_pos_weight(train_df),
        ).to(device)
        return model, optimizer, scaler, criterion

    model, optimizer, scaler, criterion = fresh_components()

    name = f"al-{args.strategy}" + (f"-{args.tag}" if args.tag else "")
    run_dir = make_run_dir(args.out, name, seed=args.seed)
    dump_json({"args": vars(args), "model": model_config(args)}, run_dir / "args.json")
    curve_logger = CSVLogger(
        run_dir / "curve.csv",
        ["round", "n_labeled", "query_seconds", "train_loss", "val_loss", "val_phase_acc",
         "val_phase_f1", "val_tool_map", "seconds"],
    )
    curve: list[dict] = []

    for rnd in range(1, args.n_rounds + 1):
        t0 = time.time()
        if args.fresh_model and rnd > 1:
            model, optimizer, scaler, criterion = fresh_components()

        n_labeled = pool.n_labeled
        batch = max(2, min(args.batch_size, n_labeled))
        labeled_loader = build_loader(
            pool.labeled_subset(), batch, True, args.workers,
            drop_last=n_labeled >= batch,
        )
        print(f"\nround {rnd}/{args.n_rounds} | labeled {n_labeled}/{pool.n_total} "
              f"| {len(labeled_loader)} batches of {batch}")

        train_loss = None
        for epoch in range(args.epochs_per_round):
            train_loss, train_acc = train_one_epoch(
                model, labeled_loader, criterion, optimizer, scaler, device, amp, args.grad_clip
            )
            print(f"  epoch {epoch + 1}/{args.epochs_per_round} "
                  f"loss {train_loss:.3f} train_acc {train_acc:.3f}")

        result = evaluate_frame_model(model, val_loader, device, criterion, amp)
        vm = scalar_metrics(result.metrics)
        print(f"  val loss {vm.get('loss', float('nan')):.3f} "
              f"acc {vm.get('phase_acc', float('nan')):.3f} "
              f"f1 {vm.get('phase_f1_macro', float('nan')):.3f} "
              f"mAP {vm.get('tool_map', float('nan')):.3f}")

        query_seconds = 0.0
        if rnd < args.n_rounds and pool.n_unlabeled > 0:
            q0 = time.time()
            selected = query(pool, model, device, amp, args)
            pool.label(selected)
            query_seconds = time.time() - q0
            print(f"  queried {len(selected)} frames -> labeled {pool.n_labeled}/{pool.n_total} "
                  f"({query_seconds:.1f}s)")

        record = {
            "round": rnd,
            "n_labeled": n_labeled,
            "query_seconds": round(query_seconds, 2),
            "train_loss": train_loss,
            "val_loss": vm.get("loss"),
            "val_phase_acc": vm.get("phase_acc"),
            "val_phase_f1": vm.get("phase_f1_macro"),
            "val_tool_map": vm.get("tool_map"),
            "seconds": round(time.time() - t0, 1),
        }
        curve.append(record)
        curve_logger.log(record)
    # ---- final artefacts --------------------------------------------------
    test_path = csv_path("test")
    final_metrics = {}
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        if args.limit:
            test_df = test_df.iloc[: args.limit]
        test_loader = build_loader(
            FrameDataset(test_df, frames_root=args.frames_dir, size=args.size),
            args.batch_size, False, args.workers,
        )
        final_metrics = scalar_metrics(evaluate_frame_model(model, test_loader, device, criterion, amp).metrics)
        print(f"test acc {final_metrics.get('phase_acc', float('nan')):.3f} "
              f"f1 {final_metrics.get('phase_f1_macro', float('nan')):.3f} "
              f"mAP {final_metrics.get('tool_map', float('nan')):.3f}")

    dump_json(
        {"strategy": args.strategy, "args": vars(args), "curve": curve,
         "final_test_metrics": final_metrics, "labeled_indices": pool.labeled_indices.tolist()},
        run_dir / f"al-{args.strategy}.json",
    )
    save_checkpoint(
        run_dir / "final.pt", model,
        extra={"model_config": model_config(args), "args": vars(args),
               "labeled_indices": pool.labeled_indices.tolist()},
    )
    print(f"\nresults: {run_dir}")
    return 0


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, grad_clip) -> tuple[float, float]:
    model.train()
    model.set_backbone_trainable(True)
    loss_m = AverageMeter()
    phase_m = PhaseMetrics()

    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        tools = batch["tools"].to(device, non_blocking=True)
        phase = batch["phase"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            out = model(x)
            loss, _ = criterion(out, tools, phase)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(float(loss.detach()), x.size(0))
        phase_m.update(out.phase_logits.argmax(-1).cpu().numpy(), batch["phase"].numpy())

    return loss_m.avg, phase_m.compute()["phase_acc"]


def query(pool: ALPool, model, device, amp, args) -> np.ndarray:
    """Run one inference pass over the unlabeled pool and pick indices."""
    unl_loader = build_loader(
        pool.unlabeled_subset(), args.query_batch_size, False, args.workers,
    )
    scores = score_loader(model, unl_loader, device, amp=amp)
    cand = CandidateSet(probs=scores["phase_probs"], feats=scores["features"])

    mc_probs = None
    if args.strategy == "mc_dropout":
        print(f"  MC-dropout: {args.mc_passes} passes over {pool.n_unlabeled} frames")
        mc_probs = mc_dropout_probs(model, unl_loader, device, passes=args.mc_passes, amp=amp)

    labeled_feats = None
    if args.strategy == "coreset":
        lab_loader = build_loader(pool.labeled_subset(), args.query_batch_size, False, args.workers)
        labeled_feats = score_loader(model, lab_loader, device, amp=amp)["features"]

    return query_indices(
        args.strategy, cand, args.query_budget,
        labeled_feats=labeled_feats, mc_probs=mc_probs,
        proj_dim=args.badge_proj_dim, seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
