"""Exactness tests for the sim2real helpers and dataset alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cholec80.data import Sim2RealDataset
from cholec80.sim2real import (
    batch_flow_and_validity,
    blend_protected,
    compute_flow,
    flow_occlusion_mask,
    warp_images,
)


def test_blend_protected_keeps_original():
    translated = torch.full((1, 3, 8, 8), -1.0)
    original = torch.full((1, 3, 8, 8), 1.0)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = 1.0                    # protect top half (H dimension)
    out = blend_protected(translated, original, mask)
    assert torch.all(out[..., :4, :] == 1.0)  # protected -> original
    assert torch.all(out[..., 4:, :] == -1.0)  # free -> translated


def test_blend_protected_gradient_blocked_on_mask():
    translated = torch.zeros(1, 3, 4, 4, requires_grad=True)
    original = torch.ones(1, 3, 4, 4)
    mask = torch.ones(1, 1, 4, 4)             # everything protected
    blend_protected(translated, original, mask).sum().backward()
    assert torch.all(translated.grad == 0)    # generator gets no signal here


def test_warp_zero_flow_is_identity():
    x = torch.rand(2, 3, 16, 24)
    flow = torch.zeros(2, 2, 16, 24)
    assert torch.allclose(warp_images(x, flow), x, atol=1e-5)


def test_warp_constant_translation():
    # Image with a vertical stripe; shifting content right by 3 px must move
    # the stripe right by 3 px (backward sampling reads at x - dx).
    x = torch.zeros(1, 1, 8, 16)
    x[..., 4] = 1.0
    flow = torch.zeros(1, 2, 8, 16)
    flow[:, 0] = 3.0                          # dx = +3 everywhere
    out = warp_images(x, flow)
    assert out[0, 0, 0, 7] == 1.0 and out[0, 0, 0, 4] == 0.0


def test_occlusion_mask_translation_is_all_valid():
    # Pure translation: forward/backward flows are exact inverses, so every
    # interior pixel passes the round-trip check.
    fw = np.zeros((32, 32, 2), dtype=np.float32)
    bw = np.zeros_like(fw)
    fw[..., 0] = 2.0
    bw[..., 0] = -2.0
    valid = flow_occlusion_mask(fw, bw)
    assert valid[2:-2, 2:-2].min() == 1.0     # interior valid
    assert valid[0].max() == 0.0              # border always invalid


def test_occlusion_mask_detects_inconsistency():
    fw = np.zeros((16, 16, 2), dtype=np.float32)
    bw = np.zeros_like(fw)
    fw[..., 0] = 3.0
    bw[..., 0] = 3.0                          # wrong sign: round trip fails
    valid = flow_occlusion_mask(fw, bw)
    assert valid[2:-2, 2:-2].max() == 0.0


def test_compute_flow_on_shifted_pattern():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    shifted = np.roll(base, 2, axis=1)        # content moves right by 2 px
    flow = compute_flow(base, shifted)
    # Interior mean dx should be close to +2 (Farneback is approximate).
    assert abs(float(flow[8:-8, 8:-8, 0].mean()) - 2.0) < 0.5


def test_batch_flow_shapes():
    prev = torch.rand(2, 3, 32, 32) * 2 - 1
    nxt = torch.roll(prev, 1, dims=-1)
    flow, valid = batch_flow_and_validity(prev, nxt)
    assert flow.shape == (2, 2, 32, 32)
    assert valid.shape == (2, 1, 32, 32)
    assert valid.min() >= 0 and valid.max() <= 1


# --------------------------------------------------------------------------- #
def _write_frame(path: Path, left_white: bool = True, size=(64, 48)):
    """Frame whose LEFT half is white -- used to verify mask alignment."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if left_white:
        arr[:, : size[0] // 2] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _write_mask(path: Path, size=(64, 48)):
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[:, : size[0] // 2] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_dataset_shapes_ranges_and_temporal_pairs(tmp_path):
    a_dir, b_dir, m_dir = tmp_path / "sim", tmp_path / "real", tmp_path / "mask"
    for v in ("video01", "video02"):
        for i in range(3):
            _write_frame(a_dir / v / f"{i * 25:06d}.jpg")
            _write_mask(m_dir / v / f"{i * 25:06d}.png")
    for i in range(2):
        _write_frame(b_dir / f"{i:06d}.jpg", left_white=False)

    ds = Sim2RealDataset(a_dir, b_dir, size=32, train=False, mask_dir=m_dir, temporal=True)
    assert len(ds) == 6
    # has_next: last frame of each video has no valid successor
    assert ds.has_next == [True, True, False, True, True, False]

    item = ds[0]
    assert item["A"].shape == (3, 32, 32)
    assert item["A"].min() >= -1.0 and item["A"].max() <= 1.0
    assert item["mask_a"].shape == (1, 32, 32)
    assert set(item["mask_a"].unique().tolist()) <= {0.0, 1.0}
    assert bool(item["pair_valid"]) is True

    last = ds[2]                              # last frame of video01
    assert bool(last["pair_valid"]) is False
    assert torch.equal(last["A"], last["A_next"])


def test_dataset_mask_stays_aligned_with_image(tmp_path):
    a_dir, b_dir, m_dir = tmp_path / "sim", tmp_path / "real", tmp_path / "mask"
    _write_frame(a_dir / "000000.jpg")
    _write_mask(m_dir / "000000.png")
    _write_frame(b_dir / "000000.jpg", left_white=False)

    ds = Sim2RealDataset(a_dir, b_dir, size=32, train=True, mask_dir=m_dir)
    for _ in range(5):                        # several random crops/flips
        item = ds[0]
        white = item["A"].mean(0) > 0.9       # white pixels of the image
        mask = item["mask_a"][0] > 0.5
        # wherever the mask says "protect", the image must be the white half
        assert torch.all(white[mask])
