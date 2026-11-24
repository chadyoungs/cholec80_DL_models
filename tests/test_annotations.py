"""Annotation parsing / index building / frame extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cholec80.annotations import (
    PHASES,
    TOOLS,
    build_index,
    extract_frames,
    read_phase_annotation,
    read_tool_annotation,
    split_videos,
)


def _write_phase(path: Path, rows: list[tuple[int, str]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Frame\tPhase\n")
        for frame, phase in rows:
            fh.write(f"{frame}\t{phase}\n")


def _write_tool(path: Path, rows: list[tuple[int, list[int]]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Frame\t" + "\t".join(TOOLS) + "\n")
        for frame, values in rows:
            fh.write(f"{frame}\t" + "\t".join(str(v) for v in values) + "\n")


def test_read_phase_annotation_forward_fill(tmp_path):
    path = tmp_path / "video01-phase.txt"
    _write_phase(path, [(0, " Preparation "), (2, PHASES[1]), (5, PHASES[2])])
    labels = read_phase_annotation(path)
    assert labels.dtype == np.int16
    assert labels.tolist() == [0, 0, 1, 1, 1, 2]


def test_read_tool_annotation_dense(tmp_path):
    path = tmp_path / "video01-tool.txt"
    values = [1, 0, 0, 0, 0, 0, 0]
    _write_tool(path, [(0, values), (25, [0, 1, 0, 0, 0, 0, 0])])
    frames, matrix = read_tool_annotation(path)
    assert frames.tolist() == [0, 25]
    assert matrix.shape == (2, 7)
    assert matrix[1, 1] == 1 and matrix[1, 0] == 0


def test_build_index_filters_unknown_and_uses_relative_paths(tmp_path):
    phase_dir, tool_dir = tmp_path / "phase", tmp_path / "tool"
    phase_dir.mkdir()
    tool_dir.mkdir()
    # frames 0-1 undefined (leading gap), then 2..9 defined
    _write_phase(phase_dir / "video01-phase.txt", [(2, PHASES[0]), (6, PHASES[1])])
    _write_tool(tool_dir / "video01-tool.txt", [(0, [1, 0, 0, 0, 0, 0, 0]), (25, [0, 1, 0, 0, 0, 0, 0])])

    df = build_index(["video01"], phase_dir, tool_dir, stride=2)
    assert list(df.columns) == ["video", "frame", "path", "phase", *TOOLS]
    assert df["frame"].tolist() == [2, 4, 6]      # 7 annotated frames (0..6), stride 2
    assert df["phase"].tolist() == [0, 0, 1]
    # 1-fps tool annotation is forward-filled to every sampled frame
    assert df[TOOLS[0]].tolist() == [1, 1, 1]
    # portable paths
    assert df["path"].iloc[0] == "video01/000002.jpg"


def test_split_videos_is_disjoint_and_ordered():
    videos = [f"video{i:02d}" for i in range(1, 11)]
    split = split_videos(videos, test_ratio=0.5, val_ratio=0.2, seed=0, protocol="ordered")
    assert split["train"] + split["val"] + split["test"] == sorted(videos)
    assert len(split["test"]) == 5
    assert set(split["test"]).isdisjoint(split["train"])
    assert set(split["val"]).isdisjoint(split["test"])


def test_split_videos_random_is_reproducible():
    videos = [f"video{i:02d}" for i in range(1, 11)]
    a = split_videos(videos, 0.5, 0.1, seed=7, protocol="random")
    b = split_videos(videos, 0.5, 0.1, seed=7, protocol="random")
    assert a == b


def _write_video(path: Path, n_frames: int = 30, size=(64, 48)) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 25, size)
    assert writer.isOpened(), "test environment lacks an MJPG encoder"
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (i * 7) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_extract_frames_stride_and_resize(tmp_path):
    video = _write_video(tmp_path / "video01.avi")
    out = tmp_path / "frames" / "video01"
    n = extract_frames(video, out, stride=10, size=32, crop="none")
    assert n == 3
    files = sorted(out.glob("*.jpg"))
    assert [f.name for f in files] == ["000000.jpg", "000010.jpg", "000020.jpg"]
    img = cv2.imread(str(files[0]))
    assert img.shape[:2] == (32, 32)


def test_extract_frames_is_idempotent(tmp_path):
    video = _write_video(tmp_path / "video01.avi", n_frames=20)
    out = tmp_path / "frames"
    assert extract_frames(video, out, stride=5, size=16) == 4
    assert extract_frames(video, out, stride=5, size=16) == 4  # existing files are skipped
