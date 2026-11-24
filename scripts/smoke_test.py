#!/usr/bin/env python
"""End-to-end smoke test on a *synthetic* mini-Cholec80 (CPU, ~2 minutes).

It writes tiny videos + annotations, then runs the real CLIs:

    prepare_dataset -> train_endonet -> evaluate -> train_active -> train_slowfast

Use this to verify an environment before downloading the real dataset::

    python scripts/smoke_test.py            # full
    python scripts/smoke_test.py --skip-slowfast
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PHASE_COLORS = [
    (60, 60, 200), (60, 200, 60), (200, 60, 60), (60, 200, 200), (200, 160, 60),
]
PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
]
TOOL_ORDER = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


def _open_writer(path: Path, size: tuple[int, int], fps: int):
    for suffix, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        writer = cv2.VideoWriter(str(path.with_suffix(suffix)),
                                 cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path.with_suffix(suffix)
        writer.release()
    raise RuntimeError("OpenCV cannot create a synthetic video (no mp4v/MJPG encoder)")


def synthesize(root: Path, n_videos: int = 6, n_frames: int = 300, size=(256, 256), fps: int = 25):
    videos_dir = root / "videos"
    phase_dir = root / "phase_annotations"
    tool_dir = root / "tool_annotations"
    for d in (videos_dir, phase_dir, tool_dir):
        d.mkdir(parents=True, exist_ok=True)

    segment = n_frames // len(PHASE_NAMES)
    for v in range(1, n_videos + 1):
        name = f"video{v:02d}"
        writer, path = _open_writer(videos_dir / f"{name}.mp4", size, fps)
        rng = np.random.default_rng(v)

        phase_rows, tool_rows = [], []
        for f in range(n_frames):
            phase_id = min(f // segment, len(PHASE_NAMES) - 1)
            frame = np.full((size[1], size[0], 3), PHASE_COLORS[phase_id], dtype=np.uint8)
            frame = np.clip(frame + rng.integers(-12, 12, frame.shape, dtype=np.int16), 0, 255).astype(np.uint8)

            # moving "instrument": a bright bar whose colour depends on the phase
            x = int(20 + (size[0] - 80) * (f % segment) / max(1, segment - 1))
            y = 40 + 10 * phase_id
            colour = (255, 255, 255) if phase_id % 2 == 0 else (255, 220, 120)
            cv2.rectangle(frame, (x, y), (x + 40, y + 18), colour, -1)
            writer.write(frame)

            phase_rows.append((f, PHASE_NAMES[phase_id]))
            if f % fps == 0:
                tools = [1 if phase_id in (0, 1, 2) else 0,      # Grasper
                         1 if phase_id in (1, 2, 3) else 0,      # Bipolar
                         1 if phase_id in (3, 4) else 0,         # Hook
                         0, 0, 0, 0]
                tool_rows.append((f, tools))
        writer.release()

        with (phase_dir / f"{name}-phase.txt").open("w", encoding="utf-8") as fh:
            fh.write("Frame\tPhase\n")
            for frame, phase in phase_rows:
                fh.write(f"{frame}\t{phase}\n")

        with (tool_dir / f"{name}-tool.txt").open("w", encoding="utf-8") as fh:
            fh.write("Frame\t" + "\t".join(TOOL_ORDER) + "\n")
            for frame, tools in tool_rows:
                fh.write(f"{frame}\t" + "\t".join(str(t) for t in tools) + "\n")

    print(f"synthetic dataset: {n_videos} videos x {n_frames} frames -> {root}")


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--work-dir", default="runs/smoke")
    p.add_argument("--skip-slowfast", action="store_true")
    p.add_argument("--clean", action="store_true", help="remove the work dir first")
    args = p.parse_args(argv)

    work = ROOT / args.work_dir
    if args.clean and work.exists():
        shutil.rmtree(work)
    data_dir = work / "data"
    if not (data_dir / "videos").exists():
        synthesize(data_dir)

    syn = data_dir
    index = work / "index"
    frames = work / "frames"
    out = work / "runs"
    py = sys.executable

    run([py, "scripts/prepare_dataset.py", "--root", syn, "--frames-dir", frames,
         "--index-dir", index, "--stride", 25, "--size", 64, "--crop", "square", "--workers", 1])

    common = ["--index-dir", index, "--frames-dir", frames, "--backbone", "resnet18",
              "--no-pretrained", "--size", 64, "--workers", 0, "--no-amp", "--seed", 0]
    run([py, "scripts/train_endonet.py", *common, "--epochs", 2, "--batch-size", 8,
         "--freeze-epochs", 0, "--patience", 0, "--out", out, "--name", "smoke-endonet"])

    ckpt = sorted(out.glob("smoke-endonet_*/checkpoints/best.pt"))[-1]
    run([py, "scripts/evaluate.py", "--checkpoint", ckpt, "--index-dir", index,
         "--frames-dir", frames, "--batch-size", 8, "--workers", 0, "--no-amp"])

    run([py, "scripts/train_active.py", *common, "--strategy", "random", "--n-rounds", 2,
         "--query-budget", 8, "--initial-ratio", 0.2, "--epochs-per-round", 1,
         "--batch-size", 8, "--out", out, "--tag", "smoke"])

    if not args.skip_slowfast:
        run([py, "scripts/train_slowfast.py", "--index-dir", index, "--frames-dir", frames,
             "--size", 64, "--clip-len", 16, "--tau", 4, "--alpha", 2, "--clip-stride", 8,
             "--slow-channels", 8, "--beta-inv", 2, "--epochs", 1, "--batch-size", 2,
             "--workers", 0, "--no-amp", "--no-inflate", "--patience", 0,
             "--out", out, "--name", "smoke-slowfast"])

    produced = sorted(str(p.relative_to(ROOT)) for p in work.rglob("*") if p.is_file())
    print("\nsmoke test finished; key artefacts:")
    for path in produced:
        if any(key in path for key in ("metrics.csv", "best.pt", "metrics.json", "curve.csv", "confusion.csv")):
            print(f"  {path}")
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
