"""Clip dataset for SlowFast: one sample yields both temporal views of a clip."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..constants import TOOLS
from .transforms import build_transform


class ClipDataset(Dataset):
    """Clip of ``clip_len`` consecutive sampled frames, seen at two rates.

    ``fast`` contains ``T_fast = clip_len // fast_step`` frames, ``slow`` contains
    ``T_slow = T_fast // (tau // alpha)`` frames which are a *subset* of the fast
    frames (so every JPEG is decoded once).

    Returns ``(slow[T_slow,3,H,W], fast[T_fast,3,H,W], tools, phase)`` where the
    labels come from the clip's centre frame. :func:`collate_clips` transposes
    them to the ``(B,C,T,H,W)`` layout that ``nn.Conv3d`` expects.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        frames_root: str | Path,
        size: int = 224,
        clip_len: int = 64,
        tau: int = 8,
        alpha: int = 4,
        clip_stride: int = 32,
        train: bool = False,
        augment: bool = False,
        transform=None,
        limit: int | None = None,
    ):
        if tau % alpha != 0:
            raise ValueError("tau must be a multiple of alpha (SlowFast convention)")
        self.fast_step = tau // alpha
        self.tau = tau
        self.alpha = alpha
        if clip_len < self.fast_step:
            raise ValueError("clip_len too small for the requested tau/alpha")

        df = index.sort_values(["video", "frame"]).reset_index(drop=True)
        self.paths = [str(Path(frames_root) / p) for p in df["path"].tolist()]
        self.tools = np.array(df[list(TOOLS)].to_numpy(dtype=np.float32), copy=True)
        self.phase = np.array(df["phase"].to_numpy(dtype=np.int64), copy=True)
        self.transform = transform or build_transform(size, train=train, augment=augment)

        self.samples: list[tuple[int, int]] = []
        video = df["video"].to_numpy()
        bounds = np.flatnonzero(video[1:] != video[:-1]) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(df)]])

        # Clips must have identical shapes to collate, but short videos (e.g. a 1 fps
        # index of a 12 s clip) cannot fill clip_len: shrink the window to the longest
        # video and keep it aligned to fast_step so every clip has the same T_fast.
        longest = int((ends - starts).max()) if len(starts) else 0
        effective = (min(clip_len, longest) // self.fast_step) * self.fast_step
        if effective < self.fast_step:
            raise ValueError(
                f"no video is long enough for a clip ({longest} frames available, "
                f"need at least {self.fast_step}); build a denser index or lower --tau"
            )
        if effective != clip_len:
            warnings.warn(
                f"clip_len reduced from {clip_len} to {effective} to fit the longest "
                f"video ({longest} sampled frames)",
                stacklevel=2,
            )
        self.clip_len = effective
        self.n_fast = effective // self.fast_step
        self.n_slow = -(-self.n_fast // self.alpha)   # ceil, matches fast[::alpha]
        for start, end in zip(starts, ends):
            if end - start < self.clip_len:
                continue
            for a in range(start, end - self.clip_len + 1, clip_stride):
                self.samples.append((a, a + self.clip_len))
        if not self.samples:
            raise ValueError("no clips could be built; check clip_stride / clip_len against the index")
        if limit:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        start, end = self.samples[idx]
        rows = np.arange(start, end)
        fast_rows = rows[:: self.fast_step]

        fast = []
        for i in fast_rows:
            with Image.open(self.paths[i]) as img:
                fast.append(self.transform(img.convert("RGB")))
        fast = torch.stack(fast, dim=0)                       # (T_fast,3,H,W)

        slow_sel = torch.arange(0, len(fast), self.tau // self.fast_step)
        slow = fast[slow_sel]                                  # (T_slow,3,H,W)

        centre = rows[len(rows) // 2]
        return (
            slow,
            fast,
            torch.from_numpy(self.tools[centre]),
            int(self.phase[centre]),
        )


def collate_clips(batch: Sequence[tuple]) -> dict:
    slow, fast, tools, phase = zip(*batch)
    # (B,T,C,H,W) -> (B,C,T,H,W) for 3-D convolutions
    slow = torch.stack(slow, dim=0).permute(0, 2, 1, 3, 4).contiguous()
    fast = torch.stack(fast, dim=0).permute(0, 2, 1, 3, 4).contiguous()
    return {
        "inputs": (slow, fast),
        "tools": torch.stack(tools, dim=0),
        "phase": torch.as_tensor(phase, dtype=torch.long),
    }
