"""Frame-level and short-sequence datasets.

Both datasets consume the CSV index produced by ``cholec80_dl.annotations.build_index``.
Paths in that CSV are relative to ``frames_root``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..constants import NUM_PHASES, TOOLS
from .transforms import build_transform


def _prepare_frame(index: pd.DataFrame, frames_root: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = index.reset_index(drop=True)
    paths = [str(Path(frames_root) / p) for p in df["path"].tolist()]
    tools = np.array(df[list(TOOLS)].to_numpy(dtype=np.float32), copy=True)
    phase = np.array(df["phase"].to_numpy(dtype=np.int64), copy=True)
    return paths, tools, phase


class FrameDataset(Dataset):
    """One sample = one frame: ``(image, tools[7], phase)``."""

    def __init__(
        self,
        index: pd.DataFrame,
        frames_root: str | Path,
        size: int = 224,
        train: bool = False,
        augment: bool = False,
        transform=None,
        limit: int | None = None,
    ):
        if limit:
            index = index.iloc[:limit]
        self.paths, self.tools, self.phase = _prepare_frame(index, frames_root)
        self.transform = transform or build_transform(size, train=train, augment=augment)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            x = self.transform(img.convert("RGB"))
        return x, torch.from_numpy(self.tools[idx]), int(self.phase[idx])

    # ------------------------------------------------------------------ #
    def phase_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for the phase head (``CrossEntropyLoss(weight=...)``)."""
        counts = np.bincount(self.phase, minlength=NUM_PHASES).astype(np.float64)
        counts[counts == 0] = counts[counts > 0].min() if (counts > 0).any() else 1.0
        weights = counts.sum() / (NUM_PHASES * counts)
        return torch.tensor(weights, dtype=torch.float32)

    @property
    def videos(self) -> list[str]:
        return sorted({Path(p).parent.name for p in self.paths})


class SequenceDataset(Dataset):
    """One sample = ``seq_len`` consecutive sampled frames of one video.

    Returns ``(clip[T,3,H,W], tools_of_last_frame, phase_of_each_frame[T])``.
    Providing the full phase sequence lets the temporal head be supervised at
    every timestep instead of only at the clip end.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        frames_root: str | Path,
        size: int = 224,
        seq_len: int = 16,
        seq_stride: int = 4,
        train: bool = False,
        augment: bool = False,
        transform=None,
        limit: int | None = None,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        df = index.sort_values(["video", "frame"]).reset_index(drop=True)
        self.paths, self.tools, self.phase = _prepare_frame(df, frames_root)
        self.transform = transform or build_transform(size, train=train, augment=augment)
        self.seq_len = seq_len

        # contiguous row ranges per video
        self.samples: list[tuple[int, int]] = []
        video = df["video"].to_numpy()
        boundaries = np.flatnonzero(video[1:] != video[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(df)]])
        for start, end in zip(starts, ends):
            if end - start < seq_len:
                continue
            for anchor in range(start + seq_len - 1, end, seq_stride):
                s = max(start, anchor - seq_len + 1)
                self.samples.append((s, s + seq_len))
        if limit:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        start, end = self.samples[idx]
        frames = []
        for i in range(start, end):
            with Image.open(self.paths[i]) as img:
                frames.append(self.transform(img.convert("RGB")))
        x = torch.stack(frames, dim=0)                       # (T,3,H,W)
        return x, torch.from_numpy(self.tools[end - 1]), torch.from_numpy(self.phase[start:end])


# --------------------------------------------------------------------------- #
# Collate functions (dict batches consumed by cholec80_dl.engine.train_step)
# --------------------------------------------------------------------------- #

def collate_frames(batch: Sequence[tuple]) -> dict:
    images, tools, phase = zip(*batch)
    return {
        "inputs": torch.stack(images, dim=0),
        "tools": torch.stack(tools, dim=0),
        "phase": torch.as_tensor(phase, dtype=torch.long),
    }


def collate_sequences(batch: Sequence[tuple]) -> dict:
    clips, tools, phases = zip(*batch)
    return {
        "inputs": torch.stack(clips, dim=0),                   # (B,T,3,H,W)
        "tools": torch.stack(tools, dim=0),                    # (B,7)
        "phase_seq": torch.stack(phases, dim=0),               # (B,T)
        "phase": torch.stack(phases, dim=0)[:, -1],            # (B,)
    }
