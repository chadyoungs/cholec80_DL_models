"""Inference helpers: batch scoring, MC-dropout passes, raw-video sliding window."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from .constants import PHASES, TOOLS
from .engine import autocast


def _unpack(out):
    """Return ``(phase_logits[B,C], tool_logits, features|None)`` for any model."""
    if hasattr(out, "phase_logits"):
        phase_logits, tool_logits, feats = out.phase_logits, out.tool_logits, getattr(out, "features", None)
    else:
        phase_logits, tool_logits = out
        feats = None
    if phase_logits.dim() == 3:          # temporal model -> last timestep
        phase_logits = phase_logits[:, -1]
    return phase_logits, tool_logits, feats


@torch.no_grad()
def score_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    """Phase probabilities + tool probabilities + features for every sample."""
    model.eval()
    phase_probs, tool_probs, feats = [], [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["inputs"].to(device, non_blocking=True)
        with autocast(device, amp):
            out = model(x)
        phase_logits, tool_logits, ft = _unpack(out)
        phase_probs.append(torch.softmax(phase_logits.float(), dim=-1).cpu().numpy())
        tool_probs.append(torch.sigmoid(tool_logits.float()).cpu().numpy())
        if ft is not None:
            feats.append(ft.float().cpu().numpy())

    result = {
        "phase_probs": np.concatenate(phase_probs, axis=0) if phase_probs else np.zeros((0, 7)),
        "tool_probs": np.concatenate(tool_probs, axis=0) if tool_probs else np.zeros((0, 7)),
    }
    if feats:
        result["features"] = np.concatenate(feats, axis=0)
    return result


@torch.no_grad()
def mc_dropout_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    passes: int = 10,
    amp: bool = True,
) -> np.ndarray:
    """``(passes, N, C)`` softmax outputs with dropout active, BatchNorm frozen."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()

    batches: list[np.ndarray] = []
    for _ in range(passes):
        probs = []
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            with autocast(device, amp):
                out = model(x)
            phase_logits, _, _ = _unpack(out)
            probs.append(torch.softmax(phase_logits.float(), dim=-1).cpu().numpy())
        batches.append(np.concatenate(probs, axis=0))

    model.eval()
    return np.stack(batches, axis=0)


def _crop(frame: np.ndarray, crop: str) -> np.ndarray:
    if crop in ("", None, "none"):
        return frame
    if crop == "square":
        h, w = frame.shape[:2]
        side = min(h, w)
        return frame[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]
    if crop.startswith("rect:"):
        x0, y0, x1, y1 = (int(v) for v in crop.split(":", 1)[1].split(","))
        return frame[y0:y1, x0:x1]
    raise ValueError(f"unknown crop mode {crop!r}")


@torch.no_grad()
def predict_video(
    model: nn.Module,
    video_path: str | Path,
    transform,
    device: torch.device,
    stride: int = 25,
    crop: str = "square",
    batch_size: int = 32,
    amp: bool = True,
) -> pd.DataFrame:
    """Sliding-window predictions on a raw video file (no frame extraction needed).

    Returns one row per sampled frame with phase probabilities, the argmax
    phase and the seven tool probabilities.
    """
    model.eval()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    pending: list[torch.Tensor] = []
    pending_frames: list[int] = []
    rows: list[dict] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(pending, dim=0).to(device)
        with autocast(device, amp):
            out = model(batch)
        phase_logits, tool_logits, _ = _unpack(out)
        probs = torch.softmax(phase_logits.float(), dim=-1).cpu().numpy()
        tools = torch.sigmoid(tool_logits.float()).cpu().numpy()
        for frame_idx, p, t in zip(pending_frames, probs, tools):
            row = {"frame": frame_idx, "phase_pred": int(p.argmax())}
            row.update({f"p_{name}": float(v) for name, v in zip(PHASES, p)})
            row.update({f"tool_{name}": float(v) for name, v in zip(TOOLS, t)})
            rows.append(row)
        pending.clear()
        pending_frames.clear()

    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                frame = _crop(frame, crop)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pending.append(transform(image))
                pending_frames.append(index)
                if len(pending) >= batch_size:
                    flush()
            index += 1
    finally:
        cap.release()
        flush()

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "video", Path(video_path).stem)
    return df
