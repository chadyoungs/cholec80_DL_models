"""Pure-numpy metrics: phase classification, tool AP, video-level Jaccard.

Keeping this dependency-free (no sklearn / torchmetrics) makes the environment
small enough for a rented GPU box and keeps results reproducible.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .constants import NUM_PHASES, NUM_TOOLS, PHASES, TOOLS


# --------------------------------------------------------------------------- #
# Frame-level phase metrics
# --------------------------------------------------------------------------- #

class PhaseMetrics:
    """Confusion-matrix accumulator for single-label phase classification."""

    def __init__(self, num_classes: int = NUM_PHASES, class_names: Sequence[str] = PHASES):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray | Sequence[int], target: np.ndarray | Sequence[int]) -> None:
        pred = np.asarray(pred, dtype=np.int64).ravel()
        target = np.asarray(target, dtype=np.int64).ravel()
        keep = (target >= 0) & (target < self.num_classes)
        np.add.at(self.cm, (target[keep], pred[keep]), 1)

    def compute(self) -> dict[str, float]:
        cm = self.cm.astype(np.float64)
        total = cm.sum()
        acc = float(np.trace(cm) / total) if total else 0.0

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        jaccard = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)

        # macro over classes present in the ground truth
        present = (cm.sum(axis=1) > 0)
        f1_macro = float(f1[present].mean()) if present.any() else 0.0
        rec_macro = float(recall[present].mean()) if present.any() else 0.0

        out = {"phase_acc": acc, "phase_f1_macro": f1_macro, "phase_recall_macro": rec_macro}
        for i, name in enumerate(self.class_names):
            out[f"phase_f1/{name}"] = float(f1[i])
            out[f"phase_recall/{name}"] = float(recall[i])
            out[f"phase_precision/{name}"] = float(precision[i])
            out[f"phase_jaccard/{name}"] = float(jaccard[i])
        return out

    def confusion(self) -> np.ndarray:
        return self.cm.copy()


# --------------------------------------------------------------------------- #
# Frame-level tool metrics (multi-label)
# --------------------------------------------------------------------------- #

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Step-wise average precision (area under the PR curve)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    positives = y_true.sum()
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / positives

    # envelope from the right, then integrate over recall steps
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.sum(np.diff(mrec) * mpre[:-1]))


class ToolMetrics:
    """Accumulates per-class scores/targets and computes AP + mAP + F1@0.5."""

    def __init__(self, num_classes: int = NUM_TOOLS, class_names: Sequence[str] = TOOLS, threshold: float = 0.5):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.threshold = threshold
        self.scores: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []

    def update(self, probs: np.ndarray, target: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 1:
            probs = probs[None, :]
        self.scores.append(probs.copy())
        self.targets.append(np.asarray(target, dtype=np.float32).reshape(probs.shape))

    def compute(self) -> dict[str, float]:
        if not self.scores:
            return {"tool_map": 0.0, "tool_f1_micro": 0.0}
        scores = np.concatenate(self.scores, axis=0)
        targets = np.concatenate(self.targets, axis=0)

        aps = []
        out: dict[str, float] = {}
        for j, name in enumerate(self.class_names):
            ap = average_precision(targets[:, j], scores[:, j])
            out[f"tool_ap/{name}"] = ap
            if not np.isnan(ap):
                aps.append(ap)
        mAP = float(np.mean(aps)) if aps else 0.0

        pred = (scores >= self.threshold).astype(np.float32)
        tp = float((pred * targets).sum())
        fp = float((pred * (1 - targets)).sum())
        fn = float(((1 - pred) * targets).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        acc = float((pred == targets).mean())

        out.update({"tool_map": mAP, "tool_f1_micro": float(f1), "tool_acc": acc})
        return out


# --------------------------------------------------------------------------- #
# Video-level phase metrics (temporal)
# --------------------------------------------------------------------------- #

def video_phase_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Frame accuracy + per-phase Jaccard for one video's frame sequence."""
    pred = np.asarray(pred, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    out: dict[str, float] = {"video_acc": float((pred == target).mean()) if len(target) else 0.0}
    for i, name in enumerate(PHASES):
        gt = target == i
        pr = pred == i
        union = np.logical_or(gt, pr).sum()
        inter = np.logical_and(gt, pr).sum()
        out[f"video_jaccard/{name}"] = float(inter / union) if union else float("nan")
    return out


def aggregate_video_metrics(per_video: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Mean over videos, ignoring nan (phase absent in that video / prediction)."""
    if not per_video:
        return {}
    keys = list(next(iter(per_video.values())).keys())
    agg: dict[str, float] = {}
    for key in keys:
        values = np.array([v[key] for v in per_video.values()], dtype=np.float64)
        finite = values[np.isfinite(values)]
        agg[f"{key}_mean"] = float(finite.mean()) if finite.size else float("nan")
    return agg
