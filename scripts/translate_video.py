#!/usr/bin/env python
"""Apply a trained sim-to-real generator to frames or a whole video.

    # image folder (e.g. simulator dumps)
    python scripts/translate_video.py \
        --checkpoint runs/sim2real_xxx/checkpoints/last.pt \
        --input data/sim/frames/video01 --out-dir output/translated \
        --mask-dir data/sim/instrument_masks/video01 --temporal-smooth 0.5

    # raw video file
    python scripts/translate_video.py --checkpoint ... \
        --input data/sim/videos/run01.mp4 --out-dir output/translated --stride 1

WHAT HAPPENS TO EACH FRAME
==========================
1. TRANSLATE   G_AB repaints organ texture into the real domain.
2. PROTECT     if a mask exists, original instrument pixels are composited
               back (bit-exact), so semantic labels stay valid.
3. DE-FLICKER  optional flow-guided temporal smoothing:

       out_t = alpha * warp(out_{t-1}, flow_{t-1->t}) + (1-alpha) * out_t

   but ONLY where the forward-backward flow check says the pixel is
   trustworthy.  Why warp instead of a naive EMA between frames?  Because
   the camera moves: averaging un-aligned frames ghosts every edge.
   Warping first re-aligns the previous result to the current geometry,
   so the averaging only removes translation noise, not motion.
   Newly (dis)occluded pixels fail the check and keep the fresh translation.

RESOLUTION NOTE
===============
The generator is fully convolutional (2 stride-2 downsamples), so any
input size divisible by 4 works -- you are NOT locked to the training
crop size.  ``--size 0`` keeps the native resolution; larger sizes cost
quadratic memory.  Masks are resized with NEAREST to stay binary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cholec80.engine import autocast  # noqa: E402
from cholec80.models import Generator  # noqa: E402
from cholec80.sim2real import blend_protected, compute_flow, flow_occlusion_mask, to_uint8_frames, warp_images  # noqa: E402
from cholec80.utils import describe_device, resolve_device, set_seed  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="train_sim2real.py checkpoint (last.pt)")
    p.add_argument("--input", required=True, help="image directory OR video file")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mask-dir", default=None,
                   help="protection masks mirroring the input layout (.png, white=keep)")
    p.add_argument("--size", type=int, default=0,
                   help="resize short side to this (0 = native resolution, must be /4)")
    p.add_argument("--stride", type=int, default=1, help="frame stride for video inputs")
    p.add_argument("--limit", type=int, default=0, help="max frames (0 = all)")
    p.add_argument("--temporal-smooth", type=float, default=0.0,
                   help="alpha in [0,1): weight of the flow-warped previous output (0 = off)")
    p.add_argument("--flow-threshold", type=float, default=1.5)
    p.add_argument("--side-by-side", action="store_true",
                   help="also write input|output comparison images")
    p.add_argument("--write-video", action="store_true",
                   help="also assemble the translated frames into an .mp4")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
def frame_source(input_path: Path, stride: int):
    """Yield (name, rgb_uint8, mask_path_or_None) in temporal order.

    Two input modes:
    * directory -> sorted images (zero-padded frame names sort correctly);
    * video     -> sequential decode, keeping every ``stride``-th frame.
    """
    if input_path.is_dir():
        files = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no images in {input_path}")
        for p in files:
            with Image.open(p) as img:
                yield p.stem, np.asarray(img.convert("RGB")), p
    elif input_path.suffix.lower() in VIDEO_SUFFIXES:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {input_path}")
        idx = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    yield f"{idx:06d}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None
                idx += 1
        finally:
            cap.release()
    else:
        raise ValueError(f"unsupported input: {input_path}")


def load_mask(mask_dir: Path | None, input_root: Path, frame_path: Path | None,
              hw: tuple[int, int]) -> np.ndarray:
    """(H,W) float32 mask, 1 = protect.  Missing -> zeros (translate everything)."""
    if mask_dir is None or frame_path is None:
        return np.zeros(hw, dtype=np.float32)
    rel = frame_path.relative_to(input_root).with_suffix(".png")
    path = mask_dir / rel
    if not path.is_file():
        return np.zeros(hw, dtype=np.float32)
    m = Image.open(path).convert("L").resize((hw[1], hw[0]), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.float32)


def to_tensor(rgb: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """uint8 RGB -> (1,3,H,W) float in [-1,1], optionally resized (short side)."""
    img = Image.fromarray(rgb)
    if size > 0:
        img = TF.resize(img, size, TF.InterpolationMode.BICUBIC)
    w, h = img.size
    # Fully-conv net with 2x stride-2 downsampling: pad-crop to multiples of 4.
    img = img.crop((0, 0, w - w % 4, h - h % 4))
    t = TF.normalize(TF.to_tensor(img), [0.5] * 3, [0.5] * 3)
    return t.unsqueeze(0).to(device)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.temporal_smooth < 1.0:
        raise ValueError("--temporal-smooth must be in [0, 1)")
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    print(f"device: {describe_device(device)} | temporal-smooth: {args.temporal_smooth}")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_blocks = int(state.get("model_config", {}).get("n_blocks", state.get("args", {}).get("n_blocks", 9)))
    g_ab = Generator(3, 3, n_blocks=n_blocks).to(device).eval()
    g_ab.load_state_dict(state["g_ab"])
    print(f"loaded {args.checkpoint} (epoch {state.get('epoch', '?')}, n_blocks={n_blocks})")

    input_path = Path(args.input)
    input_root = input_path if input_path.is_dir() else input_path.parent
    mask_root = Path(args.mask_dir) if args.mask_dir else None
    out_dir = Path(args.out_dir)
    (out_dir / "translated").mkdir(parents=True, exist_ok=True)
    if args.side_by_side:
        (out_dir / "compare").mkdir(parents=True, exist_ok=True)

    writer = None
    prev_rgb: np.ndarray | None = None       # previous INPUT frame (for flow)
    prev_out: torch.Tensor | None = None     # previous OUTPUT tensor [-1,1]
    n_done = 0

    for name, rgb, frame_path in frame_source(input_path, args.stride):
        if args.limit and n_done >= args.limit:
            break
        x = to_tensor(rgb, args.size, device)
        mask_np = load_mask(mask_root, input_root, frame_path, x.shape[-2:])
        mask = torch.from_numpy(mask_np)[None, None].to(device)

        with torch.no_grad(), autocast(device, amp):
            out = blend_protected(g_ab(x), x, mask)

        # ---------------- flow-guided temporal smoothing -----------------
        if args.temporal_smooth > 0 and prev_rgb is not None and prev_out is not None:
            cur_rgb = to_uint8_frames(x)[0]
            if cur_rgb.shape == prev_rgb.shape:
                fw = compute_flow(prev_rgb, cur_rgb)
                bw = compute_flow(cur_rgb, prev_rgb)
                valid = flow_occlusion_mask(fw, bw, args.flow_threshold)
                flow_t = torch.from_numpy(fw.transpose(2, 0, 1))[None].to(device)
                valid_t = torch.from_numpy(valid)[None, None].to(device)
                warped_prev = warp_images(prev_out.float(), flow_t)
                # Blend only where the previous pixel is still visible:
                # alpha * aligned-history + (1-alpha) * fresh translation.
                a = args.temporal_smooth * valid_t
                out = a * warped_prev + (1.0 - a) * out.float()
                # Instruments must remain bit-exact even after smoothing.
                out = blend_protected(out, x, mask)

        prev_rgb = to_uint8_frames(x)[0]
        prev_out = out.detach()

        out_np = to_uint8_frames(out)[0]
        Image.fromarray(out_np).save(out_dir / "translated" / f"{name}.jpg", quality=95)
        if args.side_by_side:
            h, w = out_np.shape[:2]
            inp_resized = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BICUBIC))
            Image.fromarray(np.concatenate([inp_resized, out_np], axis=1)).save(
                out_dir / "compare" / f"{name}.jpg", quality=92)
        if args.write_video:
            if writer is None:
                fps = 25.0 / max(1, args.stride)
                writer = cv2.VideoWriter(str(out_dir / "translated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (out_np.shape[1], out_np.shape[0]))
            writer.write(cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))

        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done} frames ...")

    if writer is not None:
        writer.release()
    print(f"done: {n_done} frames -> {out_dir / 'translated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
