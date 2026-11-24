#!/usr/bin/env python
"""Evaluate a trained checkpoint on an index split, and/or run a raw video.

    python scripts/evaluate.py --checkpoint runs/endonet_*/checkpoints/best.pt --split test
    python scripts/evaluate.py --checkpoint ... --video data/cholec80/videos/video41.mp4

Writes ``metrics.json``, ``predictions.csv`` and ``confusion.csv`` into
``--out`` (defaults to a sibling ``eval/`` directory of the checkpoint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cholec80_dl.constants import PHASES  # noqa: E402
from cholec80_dl.data import ClipDataset, FrameDataset, build_transform, collate_clips, collate_frames  # noqa: E402
from cholec80_dl.evaluation import evaluate_frame_model, scalar_metrics  # noqa: E402
from cholec80_dl.inference import predict_video, score_loader  # noqa: E402
from cholec80_dl.metrics import aggregate_video_metrics, video_phase_metrics  # noqa: E402
from cholec80_dl.models import EndoNet, SlowFast, TemporalEndoNet  # noqa: E402
from cholec80_dl.utils import describe_device, dump_json, resolve_device  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--frames-dir", default="data/frames")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--csv", default=None, help="evaluate a custom index CSV instead of a split")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--video", nargs="*", default=None,
                   help="raw video file(s) to run sliding-window inference on")
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--crop", default="square")
    return p.parse_args(argv)


def build_model(cfg: dict) -> torch.nn.Module:
    if cfg.get("arch") == "slowfast":
        return SlowFast(
            slow_channels=cfg.get("slow_channels", 64),
            beta_inv=cfg.get("beta_inv", 8),
            alpha=cfg.get("alpha", 4),
            dropout=cfg.get("dropout", 0.5),
            input_size=cfg.get("input_size", 224),
        )
    frame = EndoNet(
        backbone=cfg.get("backbone", "resnet50"),
        pretrained=False,                     # weights come from the checkpoint
        input_size=cfg.get("input_size", 224),
        hidden=cfg.get("hidden", 512),
        dropout=cfg.get("dropout", 0.3),
        use_tool_for_phase=cfg.get("use_tool_for_phase", True),
    )
    if cfg.get("temporal", "none") != "none":
        frame = TemporalEndoNet(
            frame,
            hidden=cfg.get("rnn_hidden", 256),
            num_layers=cfg.get("rnn_layers", 2),
            cell=cfg.get("temporal", "lstm"),
            bidirectional=cfg.get("bidirectional", True),
            dropout=cfg.get("dropout", 0.3),
        )
    return frame


def main(argv=None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, best {ckpt.get('best', '?')})")
    print(f"device: {describe_device(device)} | arch: {cfg.get('arch', 'endonet')}")

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- frame / clip evaluation -----------------------------------------
    metrics = {}
    predictions = None
    index_df = None
    if args.csv is not None:
        csv_path = Path(args.csv)
    elif args.video:
        csv_path = None
    else:
        csv_path = Path(args.index_dir) / f"{args.split}.csv"
    if csv_path is not None and csv_path.exists():
        index_df = pd.read_csv(csv_path)
        if cfg.get("arch") == "slowfast":
            ds = ClipDataset(
                index_df, frames_root=args.frames_dir, size=cfg.get("input_size", 224),
                clip_len=cfg.get("clip_len", 64), tau=cfg.get("tau", 8),
                alpha=cfg.get("alpha", 4), clip_stride=cfg.get("clip_stride", 32),
            )
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True, collate_fn=collate_clips)
            result = evaluate_frame_model(model, loader, device, amp=amp)
            metrics = scalar_metrics(result.metrics)
        else:
            ds = FrameDataset(index_df, frames_root=args.frames_dir, size=cfg.get("input_size", 224))
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True, collate_fn=collate_frames)
            result = evaluate_frame_model(
                model, loader, device, amp=amp, temporal=cfg.get("temporal", "none") != "none",
                max_batches=args.max_batches or None,
            )
            metrics = scalar_metrics(result.metrics)

            scores = score_loader(model, loader, device, amp=amp, max_batches=args.max_batches or None)
            n = min(len(index_df), len(scores["phase_probs"]))
            predictions = index_df.iloc[:n][["video", "frame", "phase"]].copy()
            predictions["phase_pred"] = scores["phase_probs"][:n].argmax(axis=1)
            predictions["phase_name"] = [PHASES[i] for i in predictions["phase_pred"]]
            for j, name in enumerate(PHASES):
                predictions[f"p_{name}"] = scores["phase_probs"][:n, j]

            per_video = {}
            for video, group in predictions.groupby("video"):
                per_video[video] = video_phase_metrics(
                    group["phase_pred"].to_numpy(), group["phase"].to_numpy()
                )
            metrics.update(aggregate_video_metrics(per_video))
            predictions.to_csv(out_dir / "predictions.csv", index=False)

        if result.confusion is not None:
            cm = pd.DataFrame(result.confusion, index=list(PHASES), columns=list(PHASES))
            cm.to_csv(out_dir / "confusion.csv")

        dump_json({"checkpoint": str(args.checkpoint), "csv": str(csv_path),
                   "metrics": metrics}, out_dir / "metrics.json")
        print("\nframe-level metrics")
        for key in ("phase_acc", "phase_f1_macro", "tool_map", "tool_f1_micro", "loss"):
            if key in metrics:
                print(f"  {key:16s} {metrics[key]:.4f}")
        if "video_acc_mean" in metrics:
            jac = [v for k, v in metrics.items()
                   if k.startswith("video_jaccard/") and np.isfinite(v)]
            line = f"\nper-video: acc {metrics['video_acc_mean']:.4f}"
            if jac:
                line += f" | mean jaccard {np.mean(jac):.4f}"
            print(line)
        print(f"outputs -> {out_dir}")

    # ---- raw video inference ---------------------------------------------
    if args.video:
        if cfg.get("arch") == "slowfast":
            raise SystemExit("raw-video inference is supported for frame models only; "
                             "extract frames + index to evaluate SlowFast")
        transform = build_transform(cfg.get("input_size", 224))
        for path in args.video:
            print(f"inference on {path} ...")
            df = predict_video(
                model, path, transform, device, stride=args.stride, crop=args.crop,
                batch_size=args.batch_size, amp=amp,
            )
            target = out_dir / f"pred_{Path(path).stem}.csv"
            df.to_csv(target, index=False)
            print(f"  {len(df)} frames -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
