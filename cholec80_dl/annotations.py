"""Cholec80 annotation handling: parsing, index building and frame extraction.

Expected raw layout (the official Cholec80 release)::

    cholec80/
      videos/            video01.mp4 ... video80.mp4
      phase_annotations/ video01-phase.txt      # TSV: Frame, Phase
      tool_annotations/  video01-tool.txt       # TSV: Frame, Grasper, ..., SpecimenBag @ 1 fps

The generated index is a machine-independent CSV (paths are stored relative to
``frames_root``) with one row per *sampled* frame::

    video, frame, path, phase, Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag

``phase`` is an integer class id (see :data:`cholec80_dl.constants.PHASES`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from .constants import NUM_TOOLS, PHASE_TO_IDX, PHASES, TOOLS, VIDEO_FPS

__all__ = [
    "PHASES",
    "TOOLS",
    "VideoAnnotations",
    "discover_videos",
    "resolve_video_file",
    "read_phase_annotation",
    "read_tool_annotation",
    "load_video_annotations",
    "build_index",
    "split_videos",
    "write_index",
    "extract_frames",
    "video_frame_count",
    "INDEX_COLUMNS",
]

INDEX_COLUMNS = ["video", "frame", "path", "phase", *TOOLS]
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


# --------------------------------------------------------------------------- #
# Annotation parsing
# --------------------------------------------------------------------------- #

def _read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a Cholec80 TSV robustly (BOM, stray whitespace, no header)."""
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    # Official files have a header; some re-exports do not.
    first_cell = str(df.iloc[0, 0]).strip()
    if not first_cell.replace(".", "", 1).isdigit():
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", header=None, dtype=str)
        df.columns = [f"c{i}" for i in range(df.shape[1])]
        return df
    return df


def read_phase_annotation(path: str | Path) -> np.ndarray:
    """Return one phase class id per video frame (``-1`` where undefined).

    Annotations are given per frame; unknown/blank phases become ``-1``.
    """
    df = _read_tsv(path)
    frames = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    names = df.iloc[:, 1].astype(str).str.strip()

    valid = frames.notna()
    frames = frames[valid].astype(int).to_numpy()
    labels = np.array(
        [PHASE_TO_IDX.get(n, -1) for n in names[valid]], dtype=np.int16
    )

    n = int(frames.max()) + 1
    out = np.full(n, -1, dtype=np.int16)
    out[frames] = labels
    return _forward_fill(out, fill=-1)


def read_tool_annotation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frames, matrix)`` for the 1-fps tool annotations.

    ``matrix`` has shape ``(len(frames), NUM_TOOLS)`` with 0/1 entries.
    Missing tool columns are filled with zeros (with a warning-free fallback).
    """
    df = _read_tsv(path)
    frames = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    valid = frames.notna()
    frames = frames[valid].astype(int).to_numpy()

    matrix = np.zeros((len(frames), NUM_TOOLS), dtype=np.int8)
    for j, tool in enumerate(TOOLS):
        if tool in df.columns:
            values = pd.to_numeric(df.loc[valid, tool], errors="coerce").fillna(0)
            matrix[:, j] = (values.to_numpy() > 0).astype(np.int8)
    return frames, matrix


def _forward_fill(values: np.ndarray, fill=0) -> np.ndarray:
    """Forward-fill a 1-D array, leaving leading gaps set to ``fill``."""
    n = len(values)
    if n == 0:
        return values
    src = values
    seen = src != fill
    idx = np.where(seen, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.full(n, fill, dtype=src.dtype)
    valid = idx >= 0
    out[valid] = src[idx[valid]]
    return out


@dataclass
class VideoAnnotations:
    """Per-frame annotations of one video."""

    video: str
    phase: np.ndarray            # (n_frames,) int16, -1 = undefined
    tools: np.ndarray            # (n_frames, 7) int8
    n_frames: int


def load_video_annotations(
    video: str,
    phase_dir: str | Path,
    tool_dir: str | Path | None = None,
) -> VideoAnnotations:
    phase = read_phase_annotation(Path(phase_dir) / f"{video}-phase.txt")
    n = len(phase)

    tools = np.zeros((n, NUM_TOOLS), dtype=np.int8)
    if tool_dir is not None:
        tool_path = Path(tool_dir) / f"{video}-tool.txt"
        if tool_path.exists():
            frames, matrix = read_tool_annotation(tool_path)
            n_tool = min(n, int(frames.max()) + 1) if len(frames) else 0
            dense = np.zeros((max(n_tool, 0), NUM_TOOLS), dtype=np.int8)
            if n_tool:
                dense[frames[frames < n_tool]] = matrix[frames < n_tool]
            for j in range(NUM_TOOLS):
                dense[:, j] = _forward_fill(dense[:, j], fill=0)
            tools[: len(dense)] = dense
    return VideoAnnotations(video=video, phase=phase, tools=tools, n_frames=n)


def resolve_video_file(videos_dir: str | Path, video: str) -> Path:
    """Return the existing video file for a stem (any supported container)."""
    for suffix in VIDEO_SUFFIXES:
        candidate = Path(videos_dir) / f"{video}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no video file for {video!r} in {videos_dir}")


def discover_videos(videos_dir: str | Path) -> list[str]:
    """Return sorted video stems found in ``videos_dir``."""
    vids = sorted(
        p.stem for p in Path(videos_dir).iterdir() if p.suffix.lower() in VIDEO_SUFFIXES
    )
    return vids


# --------------------------------------------------------------------------- #
# Index building
# --------------------------------------------------------------------------- #

def build_index(
    videos: Sequence[str],
    phase_dir: str | Path,
    tool_dir: str | Path | None,
    stride: int = 25,
    path_template: str = "{video}/{frame:06d}.jpg",
    strict: bool = False,
) -> pd.DataFrame:
    """Build a frame-level index for ``videos`` sampled every ``stride`` frames.

    ``path`` values are relative to the frames root so the CSV stays portable.
    Frames whose phase label is undefined are dropped.
    """
    rows: list[pd.DataFrame] = []
    for video in videos:
        try:
            ann = load_video_annotations(video, phase_dir, tool_dir)
        except FileNotFoundError:
            if strict:
                raise
            print(f"[warn] no phase annotation for {video}: skipped")
            continue
        if ann.n_frames == 0:
            continue

        frames = np.arange(0, ann.n_frames, stride, dtype=np.int64)
        phases = ann.phase[frames]
        keep = phases >= 0
        frames, phases = frames[keep], phases[keep]

        data = {
            "video": np.repeat(video, len(frames)),
            "frame": frames,
            "path": [path_template.format(video=video, frame=int(f)) for f in frames],
            "phase": phases.astype(np.int16),
        }
        for j, tool in enumerate(TOOLS):
            data[tool] = ann.tools[frames, j].astype(np.int8)
        rows.append(pd.DataFrame(data))

    if not rows:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    df = pd.concat(rows, ignore_index=True)
    return df[INDEX_COLUMNS]


def write_index(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_index(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_videos(
    videos: Iterable[str],
    test_ratio: float = 0.5,
    val_ratio: float = 0.1,
    seed: int = 42,
    protocol: str = "ordered",
) -> dict[str, list[str]]:
    """Split *by video* (never by frame) into train / val / test.

    ``protocol="ordered"`` reproduces the common Cholec80 protocol
    (videos 1-40 train, 41-80 test). ``protocol="random"`` shuffles with ``seed``.

    ``val_ratio`` is a fraction *of the training videos* carved out for model
    selection, so the test set stays untouched by active learning / early stopping.
    """
    if protocol not in {"ordered", "random"}:
        raise ValueError(f"protocol must be 'ordered' or 'random', got {protocol!r}")

    vids = sorted(videos)
    if protocol == "random":
        rng = random.Random(seed)
        vids = vids.copy()
        rng.shuffle(vids)

    n = len(vids)
    if n < 2:
        raise ValueError("need at least 2 videos to split")

    n_test = max(1, int(round(n * test_ratio)))
    n_test = min(n_test, n - 1)
    trainval, test = vids[: n - n_test], vids[n - n_test :]

    n_val = int(round(len(trainval) * val_ratio))
    if val_ratio > 0:
        n_val = max(1, n_val)
        n_val = min(n_val, len(trainval) - 1)
    train = trainval[: len(trainval) - n_val] if n_val else trainval
    val = trainval[len(trainval) - n_val :] if n_val else []
    if not train:  # degenerate tiny dataset
        train, val = trainval[:1], trainval[1:]
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------- #
# Frame extraction
# --------------------------------------------------------------------------- #

def video_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _crop_frame(frame: np.ndarray, crop: str) -> np.ndarray:
    if crop in (None, "", "none"):
        return frame
    if crop == "square":
        h, w = frame.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        return frame[y0 : y0 + side, x0 : x0 + side]
    if crop.startswith("rect:"):
        x0, y0, x1, y1 = (int(v) for v in crop.split(":", 1)[1].split(","))
        return frame[y0:y1, x0:x1]
    raise ValueError(f"unknown crop mode {crop!r} (use none|square|rect:x0,y0,x1,y1)")


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    stride: int = 25,
    size: int = 256,
    crop: str = "square",
    quality: int = 92,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> int:
    """Decode one video sequentially and write every ``stride``-th frame as JPEG.

    Returns the number of frames written. Sequential decoding is much cheaper
    than seeking: Cholec80 videos are 30-90 min at 25 fps (~0.2-2.3 M frames).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")

    written = 0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and index >= max_frames):
                break
            if index % stride == 0:
                target = out_dir / f"{index:06d}.jpg"
                if overwrite or not target.exists():
                    frame = _crop_frame(frame, crop)
                    if size and size > 0:
                        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(
                        str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
                    )
                written += 1
            index += 1
    finally:
        cap.release()
    return written


def default_stride_for_fps(fps: int = 1) -> int:
    """Frame stride that yields ``fps`` samples per second."""
    return max(1, int(round(VIDEO_FPS / fps)))
