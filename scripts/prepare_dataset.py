#!/usr/bin/env python
"""Build train/val/test frame indexes and (optionally) extract JPEG frames.

Example (server, after copying the Cholec80 release to ``data/cholec80``)::

    python scripts/prepare_dataset.py \
        --root data/cholec80 \
        --frames-dir data/frames \
        --index-dir data/index \
        --stride 25 --size 256 --crop square --workers 8

Outputs ``data/index/{train,val,test}.csv`` (machine-independent paths) and
``data/index/split.json``. Frames are written as
``data/frames/<video>/<frame:06d>.jpg``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cholec80_dl.annotations import (  # noqa: E402
    PHASES,
    TOOLS,
    build_index,
    discover_videos,
    extract_frames,
    read_index,
    resolve_video_file,
    split_videos,
    write_index,
)
from cholec80_dl.utils import dump_json, setup_logging  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="data/cholec80", help="raw Cholec80 directory")
    p.add_argument("--videos-dir", default=None, help="override <root>/videos")
    p.add_argument("--phase-dir", default=None, help="override <root>/phase_annotations")
    p.add_argument("--tool-dir", default=None, help="override <root>/tool_annotations")
    p.add_argument("--frames-dir", default="data/frames", help="destination of extracted frames")
    p.add_argument("--index-dir", default="data/index", help="destination of index CSVs")
    p.add_argument("--stride", type=int, default=25, help="frame stride (25 = 1 fps)")
    p.add_argument("--size", type=int, default=256, help="extracted frame size (0 = keep original)")
    p.add_argument("--crop", default="square", help="none | square | rect:x0,y0,x1,y1")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality")
    p.add_argument("--workers", type=int, default=4, help="parallel video decoding processes")
    p.add_argument("--no-extract", action="store_true", help="only build the index CSV files")
    p.add_argument("--overwrite-frames", action="store_true")
    p.add_argument("--protocol", choices=["ordered", "random"], default="ordered",
                   help="ordered = videos 1..N train / N+1..test (Cholec80 convention)")
    p.add_argument("--test-ratio", type=float, default=0.5)
    p.add_argument("--val-ratio", type=float, default=0.1, help="fraction of TRAIN videos used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--videos", nargs="*", default=None, help="explicit list of video stems (debugging)")
    return p.parse_args(argv)


def _extract_one(payload) -> tuple[str, int]:
    video, videos_dir, frames_dir, args = payload
    n = extract_frames(
        resolve_video_file(videos_dir, video),
        Path(frames_dir) / video,
        stride=args.stride,
        size=args.size,
        crop=args.crop,
        quality=args.quality,
        overwrite=args.overwrite_frames,
    )
    return video, n


def summarize(df, name: str) -> dict:
    phase_counts = np.bincount(df["phase"], minlength=len(PHASES)).tolist() if len(df) else [0] * len(PHASES)
    tool_counts = {t: int(df[t].sum()) for t in TOOLS} if len(df) else {t: 0 for t in TOOLS}
    return {
        "frames": int(len(df)),
        "videos": sorted(df["video"].unique().tolist()) if len(df) else [],
        "phase_counts": dict(zip(PHASES, phase_counts)),
        "tool_positive_frames": tool_counts,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()

    root = Path(args.root)
    videos_dir = Path(args.videos_dir) if args.videos_dir else root / "videos"
    phase_dir = Path(args.phase_dir) if args.phase_dir else root / "phase_annotations"
    tool_dir = Path(args.tool_dir) if args.tool_dir else root / "tool_annotations"
    if not tool_dir.is_dir():
        tool_dir = None  # tool labels become all-zero, training still works

    videos = args.videos or discover_videos(videos_dir)
    if not videos:
        raise SystemExit(f"no videos found in {videos_dir}")

    split = split_videos(
        videos,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        protocol=args.protocol,
    )
    print(f"videos: {len(videos)} -> train {len(split['train'])}, "
          f"val {len(split['val'])}, test {len(split['test'])}")

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)

    indexes = {}
    for name in ("train", "val", "test"):
        df = build_index(split[name], phase_dir, tool_dir, stride=args.stride)
        write_index(df, index_dir / f"{name}.csv")
        indexes[name] = df
        print(f"  {name:5s}: {len(df):6d} frames")

    if not args.no_extract:
        todo = [(v, videos_dir, frames_dir, args) for name in ("train", "val", "test") for v in split[name]]
        if args.workers <= 1:
            for payload in todo:
                video, n = _extract_one(payload)
                print(f"  extracted {video}: {n} frames")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_extract_one, payload) for payload in todo]
                for fut in as_completed(futures):
                    video, n = fut.result()
                    print(f"  extracted {video}: {n} frames")

        # drop rows whose JPEG is missing (annotation/video length mismatch)
        for name, df in indexes.items():
            if len(df) == 0:
                continue
            exists = np.array([(frames_dir / p).exists() for p in df["path"]])
            if not exists.all():
                print(f"  [warn] {name}: dropping {(~exists).sum()} rows with missing frames")
                indexes[name] = df[exists].reset_index(drop=True)
                write_index(indexes[name], index_dir / f"{name}.csv")

    summary = {name: summarize(indexes[name], name) for name in indexes}
    dump_json(
        {"config": vars(args), "splits": split, "summary": summary},
        index_dir / "split.json",
    )

    print("\nsummary")
    for name, info in summary.items():
        print(f"  {name:5s} {info['frames']:6d} frames, {len(info['videos'])} videos")
        if info["frames"]:
            parts = ", ".join(f"{k}:{v}" for k, v in info["phase_counts"].items())
            print(f"        phases: {parts}")

    for name in ("train", "val", "test"):
        if summary[name]["frames"] == 0:
            print(f"  [warn] split {name!r} is empty")
    print(f"\nindex -> {index_dir.resolve()}")
    if not args.no_extract:
        print(f"frames -> {frames_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
