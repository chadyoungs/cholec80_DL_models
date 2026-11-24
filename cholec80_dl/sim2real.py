"""Core building blocks for sim-to-real texture translation.

WHY THIS MODULE EXISTS
======================
A surgical simulator gives us perfect geometry, perfect labels and perfect
instrument masks -- but its organs look like plastic: no subsurface
scattering, no wet specular film, no blood, no lens haze.  A CycleGAN-style
generator can learn the *residual appearance gap* between simulated frames
(domain A) and real laparoscopic frames (domain B, e.g. Cholec80).

Naively applying an image-to-image GAN to surgery has two well-documented
failure modes, and each function below is the antidote to one of them:

1. HALLUCINATION / LABEL CORRUPTION
   The generator may repaint instruments, invent vessels, or move anatomy.
   If the translated frames are used to train a perception model, its
   ground-truth labels (which come from the simulator!) silently become
   wrong.  -> ``blend_protected`` composites the *original* pixels of every
   semantically-critical region (instruments, known blood decals, ...) back
   on top of the translation, so the network is only ever allowed to change
   the appearance of plain organ tissue.

2. TEMPORAL FLICKER
   Each frame is translated independently, so texture "boils" from frame to
   frame even when the camera barely moves.  -> ``compute_flow`` +
   ``warp_images`` + ``flow_occlusion_mask`` let us propagate the previous
   translated frame along the scene motion and reuse it, which is exactly
   the trick behind tempCycleGAN (Pfeiffer et al., "Improving Surgical
   Training Phantoms by Hyperrealism", 2019) and classic video style
   transfer.

All functions are pure tensor/numpy operations with no learned parameters,
so they can be unit-tested exactly and reused at train AND inference time.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Instrument / semantics protection
# --------------------------------------------------------------------------- #
def blend_protected(translated: torch.Tensor,
                    original: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Composite protected pixels of ``original`` over ``translated``.

    Args:
        translated: (B,3,H,W) generator output, any value range.
        original:   (B,3,H,W) untouched input frame, SAME range as translated.
        mask:       (B,1,H,W) float in {0,1}.
                    **Convention: 1 = PROTECT (keep original), 0 = translate.**
                    We use "1 = keep" because simulator segmentation masks
                    naturally mark the object (instrument) as foreground, so
                    the raw mask can be passed straight through.

    The blend is a hard per-pixel switch written as a linear interpolation:

        out = mask * original + (1 - mask) * translated

    Two properties worth internalising:

    * It is DIFFERENTIABLE w.r.t. ``translated`` everywhere the mask is 0,
      and has zero gradient where the mask is 1.  Training through this
      composite therefore tells the generator: "you get no credit and no
      blame for instrument pixels -- spend all capacity on tissue".
    * The discriminator sees the composite, i.e. real instruments sitting
      on translated tissue.  That is exactly the distribution we will emit
      at inference time, so there is no train/test mismatch.

    A soft mask (values in (0,1), e.g. a feathered edge) also works and
    produces a smooth seam; callers can blur the binary mask beforehand.
    """
    if mask.shape[-2:] != translated.shape[-2:]:
        raise ValueError(f"mask spatial shape {mask.shape[-2:]} != image {translated.shape[-2:]}")
    return mask * original + (1.0 - mask) * translated


# --------------------------------------------------------------------------- #
# 2. Optical flow between two frames (classical, CPU, no learned weights)
# --------------------------------------------------------------------------- #
def compute_flow(prev_bgr: np.ndarray, next_bgr: np.ndarray) -> np.ndarray:
    """Dense optical flow ``prev -> next`` with Farneback's algorithm.

    Args:
        prev_bgr / next_bgr: (H,W,3) uint8 images (OpenCV BGR or RGB -- the
            algorithm only looks at grayscale, so channel order is irrelevant).

    Returns:
        (H,W,2) float32 flow field; ``flow[y, x] = (dx, dy)`` means
        "the pixel at (x,y) in PREV moved to (x+dx, y+dy) in NEXT".

    WHY FARNEBACK AND NOT A LEARNED FLOW NET?
    * It runs on CPU in ~5 ms/frame at 256px -- negligible next to the GAN.
    * Simulated frames are clean (no noise, no motion blur), which is the
      easy regime for polynomial-expansion flow.
    * No extra dependency / checkpoint to ship to the rented server.

    The parameters below are the standard Farneback defaults; the important
    ones to understand:
    * pyr_scale=0.5   -- image pyramid, catches large motion coarse-to-fine
    * levels=3        -- pyramid depth
    * winsize=15      -- averaging window; larger = smoother but blurrier flow
    * poly_n=5, poly_sigma=1.1 -- local polynomial approximation of intensity
    """
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )


# --------------------------------------------------------------------------- #
# 3. Warping a (translated) image along a flow field, on GPU, differentiably
# --------------------------------------------------------------------------- #
def warp_images(images: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``images`` so that content moves according to ``flow``.

    Args:
        images: (B,C,H,W) -- e.g. the PREVIOUS translated frame.
        flow:   (B,2,H,W) -- forward flow prev->next, in PIXELS,
                channel 0 = dx (x direction), channel 1 = dy (y direction).

    Returns:
        (B,C,H,W): ``out[b,:,y,x] = images[b,:, y-flow_y, x-flow_x]``
        (bilinear sampling), i.e. the previous frame "moved into" the current
        frame's geometry.

    IMPLEMENTATION NOTES (this is where everyone gets bugs):
    * ``F.grid_sample`` uses BACKWARD sampling: for every OUTPUT pixel it
      asks "where do I read from in the INPUT?".  Forward flow prev->next
      tells us where each prev pixel WENT; reading prev at (x - dx, y - dy)
      is the standard first-order approximation of the inverse mapping.
      It is exact for pure translation and good enough for the small
      inter-frame motion of laparoscopic video.
    * grid_sample expects coordinates normalised to [-1,1]; we convert with
      the align_corners=True convention:  norm = 2*pixel/(size-1) - 1.
    * padding_mode="border" clamps out-of-frame reads to the edge pixel,
      which avoids black borders leaking into the temporal loss.
    * The whole op is differentiable w.r.t. ``images`` -- gradients flow
      back into the generator when this is used inside a training loss.
    """
    b, _, h, w = images.shape
    # Pixel grid of the OUTPUT frame: base[y,x] = (x, y).
    gy, gx = torch.meshgrid(
        torch.arange(h, device=images.device, dtype=images.dtype),
        torch.arange(w, device=images.device, dtype=images.dtype),
        indexing="ij",
    )
    base = torch.stack((gx, gy), dim=0).unsqueeze(0)          # (1,2,H,W)
    src = base - flow                                         # read position
    src_x = 2.0 * src[:, 0] / max(w - 1, 1) - 1.0             # normalise to [-1,1]
    src_y = 2.0 * src[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((src_x, src_y), dim=-1)                # (B,H,W,2)
    return F.grid_sample(images, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


# --------------------------------------------------------------------------- #
# 4. Occlusion detection: where is the flow trustworthy?
# --------------------------------------------------------------------------- #
def flow_occlusion_mask(flow_fw: np.ndarray, flow_bw: np.ndarray,
                        threshold: float = 1.5) -> np.ndarray:
    """Forward-backward consistency check -> (H,W) float32 validity mask.

    IDEA (Sundaram et al., "Dense point trajectories by GPU-accelerated
    large displacement optical flow", ECCV 2010 -- the standard check):

    Follow a pixel forward:   p' = p + fw(p)
    Then follow it backward:  p'' = p' + bw(p')
    If the pixel is visible in BOTH frames and the flow is correct,
    p'' lands back on p.  If the pixel got OCCLUDED (e.g. an instrument
    slid over the tissue) or DISOCCLUDED, the round trip fails.

    We mark a pixel invalid when |p'' - p|^2 > threshold^2 (pixels),
    plus a 2-pixel border where flow is unreliable anyway.

    Args:
        flow_fw: (H,W,2) flow prev->next.
        flow_bw: (H,W,2) flow next->prev.
        threshold: round-trip error tolerance in pixels.

    Returns:
        (H,W) float32 in {0,1}; 1 = flow trustworthy, safe to reuse the
        warped previous translation; 0 = freshly (dis)occluded, must use
        the current frame's own translation.
    """
    h, w = flow_fw.shape[:2]
    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # Position of each prev pixel inside the NEXT frame.
    px = gx + flow_fw[..., 0]
    py = gy + flow_fw[..., 1]
    # Sample the backward flow at those (fractional) positions, bilinearly.
    bw_at_p = _bilinear_sample(flow_bw, px, py)               # (H,W,2)
    # Round-trip landing position and its error.
    rx = px + bw_at_p[..., 0]
    ry = py + bw_at_p[..., 1]
    err_sq = (rx - gx) ** 2 + (ry - gy) ** 2
    valid = (err_sq < threshold ** 2).astype(np.float32)
    # Flow at the image border is extrapolated garbage; never trust it.
    valid[:2, :] = 0.0
    valid[-2:, :] = 0.0
    valid[:, :2] = 0.0
    valid[:, -2:] = 0.0
    return valid


def _bilinear_sample(field: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a (H,W,C) field at fractional (px,py) positions.

    Written by hand so the project keeps its "no scipy" promise.  This is
    the same operation as ``scipy.ndimage.map_coordinates(order=1)``.
    """
    h, w = field.shape[:2]
    x0 = np.clip(np.floor(px).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(py).astype(np.int32), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = np.clip(px - x0, 0.0, 1.0)[..., None]
    wy = np.clip(py - y0, 0.0, 1.0)[..., None]
    top = field[y0, x0] * (1 - wx) + field[y0, x1] * wx
    bot = field[y1, x0] * (1 - wx) + field[y1, x1] * wx
    return top * (1 - wy) + bot * wy


# --------------------------------------------------------------------------- #
# 5. Batch-level glue used by the training loop and the inference script
# --------------------------------------------------------------------------- #
def to_uint8_frames(images: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) float in [-1,1]  ->  (B,H,W,3) uint8, for OpenCV.

    The GAN lives in [-1,1] (tanh output range); OpenCV lives in uint8.
    This is the only place where that conversion should happen, so the
    denormalisation formula is written exactly once.
    """
    arr = images.detach().float().clamp(-1, 1).add(1.0).mul(127.5)
    return arr.round().byte().permute(0, 2, 3, 1).cpu().numpy()


def batch_flow_and_validity(prev: torch.Tensor, nxt: torch.Tensor,
                            threshold: float = 1.5
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample forward flow + occlusion mask for a whole batch.

    Args:
        prev / nxt: (B,3,H,W) float tensors in [-1,1] (consecutive sim frames).

    Returns:
        flow:  (B,2,H,W) float32 on ``prev.device`` -- forward flow, pixels.
        valid: (B,1,H,W) float32 in {0,1} -- forward-backward consistency.

    NOTE: flow is computed on CPU with numpy and is treated as a CONSTANT
    supervision signal (no gradient).  That is intentional: we do not want
    the generator to cheat by making the flow itself easier -- the frames
    it is computed from are the *inputs*, which the generator cannot change.
    """
    prev_np = to_uint8_frames(prev)
    nxt_np = to_uint8_frames(nxt)
    flows, valids = [], []
    for a, b in zip(prev_np, nxt_np):
        fw = compute_flow(a, b)
        bw = compute_flow(b, a)
        flows.append(fw.transpose(2, 0, 1))                   # (2,H,W)
        valids.append(flow_occlusion_mask(fw, bw, threshold)[None])  # (1,H,W)
    flow = torch.from_numpy(np.stack(flows)).to(prev.device)
    valid = torch.from_numpy(np.stack(valids)).to(prev.device)
    return flow, valid
