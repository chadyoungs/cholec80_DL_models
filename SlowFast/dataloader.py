"""
Clip-based video dataloader for the Cholec80 dataset.

Cholec80 (https://cholec80.pierre.is) contains 80 laparoscopic cholecystectomy
videos annotated frame-by-frame with:
  • Surgical phase   — 7 classes (see config.phase_labels)
  • Tool presence    — 7 binary indicators (see config.tool_labels)

This module groups individual frames (loaded from a CSV prepared in the same
format as EndoNet) into temporal clips and returns two sub-sampled views of
each clip:
  • slow clip — T_slow frames at coarse temporal stride (1 frame every tau)
  • fast clip — T_fast frames at fine temporal stride (1 frame every tau/alpha)

The label of a clip is derived from its centre frame.
"""

import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config


# ── Image pipeline ─────────────────────────────────────────────────────────────

def _build_transform():
    """Return the standard spatial pre-processing pipeline."""
    return transforms.Compose([
        transforms.Resize((config.img_h, config.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _load_frame(path):
    """Load a single frame as a normalised (3, H, W) tensor."""
    img = Image.open(path).crop((50, 0, 804, 480)).convert('RGB')
    return _build_transform()(img)


# ── Dataset ────────────────────────────────────────────────────────────────────

class CholecClipDataset(Dataset):
    """Clip-based Cholec80 dataset for SlowFast.

    Each sample is a tuple (slow_clip, fast_clip, phase_label, tool_labels):
      • slow_clip  — (3, T_slow, H, W) float tensor
      • fast_clip  — (3, T_fast, H, W) float tensor
      • phase_label — scalar long tensor (0-indexed class id)
      • tool_labels — (num_tools,) float tensor with binary indicators

    Args:
        split:        'train' or 'test'.
        csv_dir:      directory that contains ``<split>_set_info.csv``.
        clip_length:  number of consecutive frames in each clip window.
        T_slow:       frames sampled for the slow pathway.
        T_fast:       frames sampled for the fast pathway.
        clip_stride:  stride (in frames) between successive clip windows.
    """

    def __init__(
        self,
        split,
        csv_dir=config.CSV_DIR,
        clip_length=config.clip_length,
        T_slow=config.T_slow,
        T_fast=config.T_fast,
        clip_stride=config.clip_stride,
    ):
        if split not in ('train', 'test'):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.T_slow = T_slow
        self.T_fast = T_fast
        self.clip_length = clip_length
        self.transform = _build_transform()

        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            csv_dir,
            f'{split}_set_info.csv',
        )
        df = pd.read_csv(csv_path)
        self.clips = self._build_clips(df, clip_stride)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_clips(self, df, clip_stride):
        """Scan the frame-level CSV and return a list of clip descriptors.

        Each descriptor is a dict with keys:
          ``slow_paths`` — list[str] of T_slow frame paths
          ``fast_paths`` — list[str] of T_fast frame paths
          ``phase``      — int, centre-frame phase label
          ``tools``      — np.ndarray[float32] of shape (num_tools,)
        """
        clips = []

        # Derive video identifier from the parent directory of each frame so
        # that clips are never built across video boundaries.
        df = df.copy()
        df['_video'] = df['file_loc'].apply(lambda p: os.path.dirname(p))

        for _, group in df.groupby('_video', sort=False):
            group = group.sort_values('file_loc').reset_index(drop=True)
            n = len(group)

            for start in range(0, n - self.clip_length + 1, clip_stride):
                end = start + self.clip_length
                window = group.iloc[start:end]

                # Uniformly subsample from the window for each pathway.
                slow_step = self.clip_length // self.T_slow  # e.g. 8
                fast_step = self.clip_length // self.T_fast  # e.g. 2
                slow_idx = [i * slow_step for i in range(self.T_slow)]
                fast_idx = [i * fast_step for i in range(self.T_fast)]

                # Labels from the centre frame of the clip.
                center = window.iloc[self.clip_length // 2]
                phase = int(center[config.phase_label_in_csv])
                tools = center[config.tool_labels].values.astype(np.float32)

                clips.append({
                    'slow_paths': window.iloc[slow_idx]['file_loc'].tolist(),
                    'fast_paths': window.iloc[fast_idx]['file_loc'].tolist(),
                    'phase': phase,
                    'tools': tools,
                })

        return clips

    def _load_clip(self, paths):
        """Load a list of frame paths into a (3, T, H, W) tensor."""
        frames = [self._load_frame(p) for p in paths]
        return torch.stack(frames, dim=1)   # (3, T, H, W)

    def _load_frame(self, path):
        img = Image.open(path).crop((50, 0, 804, 480)).convert('RGB')
        return self.transform(img)

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        slow_clip = self._load_clip(clip['slow_paths'])   # (3, T_slow, H, W)
        fast_clip = self._load_clip(clip['fast_paths'])   # (3, T_fast, H, W)
        phase = torch.tensor(clip['phase'], dtype=torch.long)
        tools = torch.from_numpy(clip['tools'])           # (num_tools,)
        return slow_clip, fast_clip, phase, tools
