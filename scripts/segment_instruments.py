#!/usr/bin/env python
"""Instrument detection + segmentation with GroundingDINO + SAM.

Replaces the old ``CycleGAN/datasets/data_prepared.py`` hard-coded script with a
configurable CLI that writes per-frame masks, instrument foregrounds (ready for
CycleGAN/unpaired translation) and a JSON sidecar with boxes/scores/labels.

Requires the optional extra::

    pip install -e ".[seg]"

    python scripts/segment_instruments.py \
        --input data/cholec80/videos --out-dir data/instruments/cholec/trainA \
        --stride 25 --max-coverage 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_PROMPT = "grasper. bipolar forceps. hook. scissors. clipper. irrigator. specimen bag. needle driver."


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="video directory or image directory")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--stride", type=int, default=25, help="frame stride for videos")
    p.add_argument("--limit", type=int, default=0, help="max frames total (0 = all)")
    p.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--sam-model", default="facebook/sam-vit-base")
    p.add_argument("--box-threshold", type=float, default=0.25)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--max-coverage", type=float, default=0.7, help="drop boxes covering more than this fraction")
    p.add_argument("--min-side", type=int, default=8, help="drop boxes smaller than this (pixels)")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--save-background", action="store_true", help="also write the masked-out background")
    p.add_argument("--mask-only", action="store_true", help="skip foreground/background images")
    p.add_argument("--save-meta", action="store_true", default=True)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Frame streams
# --------------------------------------------------------------------------- #

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_frames(input_path: Path, stride: int) -> Iterator[tuple[str, int, np.ndarray]]:
    """Yield ``(source_stem, frame_index, RGB frame)`` lazily."""
    if input_path.is_dir() and any(p.suffix.lower() in VIDEO_SUFFIXES for p in input_path.iterdir()):
        for video in sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES):
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                print(f"[warn] cannot open {video}")
                continue
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % stride == 0:
                    yield video.stem, index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                index += 1
            cap.release()
        return

    if input_path.is_dir():
        for image in sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            frame = cv2.imread(str(image))
            if frame is None:
                continue
            yield image.stem, 0, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return

    raise SystemExit(f"--input must be a video or image directory: {input_path}")


# --------------------------------------------------------------------------- #
# Detection / segmentation
# --------------------------------------------------------------------------- #

def run_dino(frame_rgb, processor, model, prompt: str, device, args):
    from PIL import Image

    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    kwargs = dict(target_sizes=[image.size[::-1]])
    try:
        result = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            box_threshold=args.box_threshold, text_threshold=args.text_threshold, **kwargs,
        )[0]
    except TypeError:  # newer transformers renamed the arguments
        result = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=args.box_threshold, **kwargs,
        )[0]

    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()
    labels = result.get("text_labels", result.get("labels", []))
    if isinstance(labels, torch.Tensor):
        labels = [str(v) for v in labels.tolist()]
    return boxes, scores, labels


def filter_boxes(boxes, scores, labels, frame_shape, args):
    """Per-label NMS + drop boxes that are too large / too small."""
    import torchvision.ops as ops

    if len(boxes) == 0:
        return boxes, scores, labels

    keep_all = []
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        idx = [i for i, lab in enumerate(labels) if lab == label]
        keep = ops.nms(boxes[idx], scores[idx], iou_threshold=args.nms_iou)
        keep_all.extend(idx[int(k)] for k in keep)

    h, w = frame_shape[:2]
    area = h * w
    final = []
    for i in keep_all:
        x0, y0, x1, y1 = boxes[i].tolist()
        box_w, box_h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        if box_w < args.min_side or box_h < args.min_side:
            continue
        if box_w * box_h / area > args.max_coverage:
            continue
        final.append(i)

    idx = torch.tensor(final, dtype=torch.long)
    return boxes[idx], scores[idx], [labels[i] for i in final]


def run_sam(frame_rgb, boxes, processor, model, device) -> np.ndarray | None:
    from PIL import Image

    if len(boxes) == 0:
        return None
    image = Image.fromarray(frame_rgb)
    inputs = processor(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0]
    masks = masks.reshape(-1, *masks.shape[-2:]).numpy() > 0  # (n_boxes, H, W)
    return np.any(masks, axis=0)


def save_frame(frame_rgb, mask, name: str, out_dir: Path, args) -> None:
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)
    (out_dir / "foreground").mkdir(parents=True, exist_ok=True)
    if args.save_background:
        (out_dir / "background").mkdir(parents=True, exist_ok=True)

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "mask" / f"{name}.png"), (mask.astype(np.uint8) * 255))
    if args.mask_only:
        return

    foreground = np.zeros_like(bgr)
    foreground[mask] = bgr[mask]
    cv2.imwrite(str(out_dir / "foreground" / f"{name}.jpg"), foreground)
    if args.save_background:
        background = bgr.copy()
        background[mask] = 0
        cv2.imwrite(str(out_dir / "background" / f"{name}.jpg"), background)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection, SamModel, SamProcessor
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "transformers is required: pip install -e '.[seg]' "
            f"(original error: {exc})"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto"
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"device: {device}")

    print(f"loading {args.dino_model} ...")
    dino_processor = AutoProcessor.from_pretrained(args.dino_model)
    dino_model = GroundingDinoForObjectDetection.from_pretrained(args.dino_model).to(device).eval()

    print(f"loading {args.sam_model} ...")
    sam_processor = SamProcessor.from_pretrained(args.sam_model)
    sam_model = SamModel.from_pretrained(args.sam_model).to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    processed = 0
    for source, frame_idx, frame_rgb in iter_frames(Path(args.input), args.stride):
        if args.limit and processed >= args.limit:
            break
        name = f"{source}_{frame_idx:06d}" if frame_idx else source
        if args.skip_existing and (out_dir / "mask" / f"{name}.png").exists():
            continue

        boxes, scores, labels = run_dino(frame_rgb, dino_processor, dino_model, args.prompt, device, args)
        boxes, scores, labels = filter_boxes(boxes, scores, labels, frame_rgb.shape, args)
        mask = run_sam(frame_rgb, boxes, sam_processor, sam_model, device)

        if mask is None:
            print(f"  {name}: no instruments detected")
            processed += 1
            continue

        save_frame(frame_rgb, mask, name, out_dir, args)
        if args.save_meta:
            with meta_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "name": name,
                    "source": source,
                    "frame": int(frame_idx),
                    "boxes": boxes.tolist(),
                    "scores": scores.tolist(),
                    "labels": labels,
                }) + "\n")
        processed += 1
        print(f"  {name}: {len(boxes)} instruments -> {out_dir}")

    print(f"done: {processed} frames -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
