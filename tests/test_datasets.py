"""Dataset shapes, collates and label plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cholec80_dl.data import ClipDataset, FrameDataset, SequenceDataset, collate_clips, collate_frames, collate_sequences
from cholec80_dl.constants import TOOLS


def _make_frames(root: Path, videos: dict[str, int], size: int = 64) -> None:
    rng = np.random.default_rng(0)
    for video, n in videos.items():
        (root / video).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / video / f"{i * 25:06d}.jpg")


def _make_index(videos: dict[str, int]) -> pd.DataFrame:
    rows = []
    for video, n in videos.items():
        for i in range(n):
            row = {"video": video, "frame": i * 25, "path": f"{video}/{i * 25:06d}.jpg", "phase": i % 4}
            for t in TOOLS:
                row[t] = int((hash((video, i, t)) % 3) == 0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_frame_dataset(tmp_path):
    videos = {"video01": 6, "video02": 4}
    _make_frames(tmp_path, videos)
    ds = FrameDataset(_make_index(videos), tmp_path, size=48)

    assert len(ds) == 10
    x, tools, phase = ds[0]
    assert x.shape == (3, 48, 48) and x.dtype == torch.float32
    assert tools.shape == (7,) and tools.dtype == torch.float32
    assert isinstance(phase, int)
    assert ds.videos == ["video01", "video02"]

    batch = collate_frames([ds[0], ds[1]])
    assert batch["inputs"].shape == (2, 3, 48, 48)
    assert batch["tools"].shape == (2, 7)
    assert batch["phase"].shape == (2,)


def test_frame_dataset_class_weights(tmp_path):
    videos = {"video01": 8}
    _make_frames(tmp_path, videos)
    ds = FrameDataset(_make_index(videos), tmp_path, size=32)
    weights = ds.phase_class_weights()
    assert weights.shape == (7,)
    assert torch.all(weights > 0)


def test_sequence_dataset_windows_stay_inside_videos(tmp_path):
    videos = {"video01": 10, "video02": 6}
    _make_frames(tmp_path, videos)
    ds = SequenceDataset(_make_index(videos), tmp_path, size=32, seq_len=4, seq_stride=2)

    assert len(ds) == 6            # video01 (10 rows): anchors 3,5,7,9 ; video02 (6 rows): 3,5
    clips, tools, phases = zip(*[ds[i] for i in range(len(ds))])
    assert all(c.shape == (4, 3, 32, 32) for c in clips)
    # every window stays inside one video and phases advance frame by frame
    assert phases[0].tolist() == [0, 1, 2, 3]
    assert phases[4].tolist() == [0, 1, 2, 3]      # first window of video02
    assert tools[0].shape == (7,)

    batch = collate_sequences([ds[0], ds[1]])
    assert batch["inputs"].shape[0] == 2 and batch["inputs"].shape[1] == 4
    assert batch["phase_seq"].shape == (2, 4)
    assert batch["phase"].shape == (2,)


def test_clip_dataset_two_views(tmp_path):
    videos = {"video01": 16}
    _make_frames(tmp_path, videos)
    ds = ClipDataset(_make_index(videos), tmp_path, size=32, clip_len=8, tau=4, alpha=2, clip_stride=4)
    assert ds.n_slow == 2 and ds.n_fast == 4 and len(ds) >= 1

    slow, fast, tools, phase = ds[0]
    assert slow.shape == (2, 3, 32, 32)
    assert fast.shape == (4, 3, 32, 32)
    assert tools.shape == (7,) and isinstance(phase, int)

    batch = collate_clips([ds[0], ds[1]])
    # collate gives (B,C,T,H,W) so the tensors can go straight into nn.Conv3d
    assert batch["inputs"][0].shape == (2, 3, 2, 32, 32)
    assert batch["inputs"][1].shape == (2, 3, 4, 32, 32)
    assert batch["tools"].shape == (2, 7)


def test_clip_dataset_rejects_bad_tau(tmp_path):
    videos = {"video01": 4}
    _make_frames(tmp_path, videos)
    try:
        ClipDataset(_make_index(videos), tmp_path, clip_len=8, tau=3, alpha=2)
    except ValueError as exc:
        assert "tau" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
