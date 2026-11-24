"""Labeled / unlabeled pool over the frame-level Cholec80 training index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from ..data import FrameDataset


class ALPool:
    """Boolean labeled-mask over a training index + convenience loaders."""

    def __init__(
        self,
        index: pd.DataFrame,
        frames_root: str | Path,
        initial_ratio: float = 0.05,
        seed: int = 42,
        size: int = 224,
        augment: bool = False,
        balance_initial: bool = True,
    ):
        self.index = index.reset_index(drop=True)
        self.dataset = FrameDataset(
            self.index, frames_root=frames_root, size=size, train=True, augment=augment
        )
        n = len(self.dataset)
        rng = np.random.default_rng(seed)

        self.mask = np.zeros(n, dtype=bool)
        n_labeled = max(1, int(round(n * initial_ratio)))
        if balance_initial:
            # guarantee at least one labeled frame per phase, then fill randomly
            phase = self.dataset.phase
            present = np.unique(phase)
            for cls in present:
                candidates = np.flatnonzero(phase == cls)
                self.mask[rng.choice(candidates)] = True
            n_labeled = max(n_labeled, int(self.mask.sum()))
        remaining = np.flatnonzero(~self.mask)
        need = max(0, n_labeled - int(self.mask.sum()))
        if need:
            self.mask[rng.choice(remaining, size=min(need, len(remaining)), replace=False)] = True

    # ------------------------------------------------------------------ #
    @property
    def labeled_indices(self) -> np.ndarray:
        return np.flatnonzero(self.mask)

    @property
    def unlabeled_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.mask)

    @property
    def n_labeled(self) -> int:
        return int(self.mask.sum())

    @property
    def n_unlabeled(self) -> int:
        return int((~self.mask).sum())

    @property
    def n_total(self) -> int:
        return len(self.mask)

    def label(self, pool_indices: np.ndarray) -> None:
        """Mark ``pool_indices`` (positions in the *unlabeled* array) as labeled."""
        idx = np.asarray(pool_indices, dtype=np.int64).ravel()
        if len(idx) == 0:
            return
        self.mask[self.unlabeled_indices[idx]] = True

    def state(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_labeled": self.n_labeled,
            "n_unlabeled": self.n_unlabeled,
            "labeled_indices": self.labeled_indices.tolist(),
            "mask_seed": 42,
        }

    def load_state(self, state: dict) -> None:
        self.mask[:] = False
        self.mask[np.asarray(state["labeled_indices"], dtype=np.int64)] = True

    # ------------------------------------------------------------------ #
    def labeled_subset(self) -> Subset:
        return Subset(self.dataset, self.labeled_indices.tolist())

    def unlabeled_subset(self) -> Subset:
        return Subset(self.dataset, self.unlabeled_indices.tolist())

    def phase_weights(self):
        """Class weights computed on the *labeled* set only (avoids label leakage)."""
        labeled = self.labeled_indices
        if len(labeled) == 0:
            return self.dataset.phase_class_weights()
        phases = self.dataset.phase[labeled]
        counts = np.bincount(phases, minlength=7).astype(np.float64)
        counts[counts == 0] = counts[counts > 0].min() if (counts > 0).any() else 1.0
        w = counts.sum() / (len(counts) * counts)
        import torch

        return torch.tensor(w, dtype=torch.float32)
