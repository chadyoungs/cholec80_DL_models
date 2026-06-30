"""
Pool-based active learning dataset for cholec80.

Wraps the EndoNet CSV format (train_set_info.csv / test_set_info.csv) and
maintains a boolean labeled/unlabeled mask over all training samples.

Classes
-------
CholecBaseDataset  – read-only PyTorch Dataset (image + labels from CSV)
CholecPoolDataset  – active-learning pool manager built on top of it
"""

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# ── resolve EndoNet config (one directory up) ─────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'EndoNet'))
import config as endonet_config

_PHASE_COLS = endonet_config.phase_labels               # 7 one-hot phase columns
_TOOL_COLS  = endonet_config.tool_labels_without_NoTool  # 7 binary tool columns


# ──────────────────────────────────────────────────────────────────────────
# Read-only dataset
# ──────────────────────────────────────────────────────────────────────────

class CholecBaseDataset(Dataset):
    """
    Read-only PyTorch Dataset for cholec80 frames.

    Reads *file_loc*, phase labels (one-hot, 7 cols) and tool labels
    (binary multi-label, 7 cols) from the EndoNet CSV files.

    Returns
    -------
    (image_tensor, tool_labels_float32, phase_labels_float32)
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.data_list    = np.asarray(df['file_loc'])
        self.phase_labels = np.asarray(df[_PHASE_COLS]).astype(np.float32)  # (N, 7)
        self.tool_labels  = np.asarray(df[_TOOL_COLS]).astype(np.float32)   # (N, 7)

    # ------------------------------------------------------------------

    def _transform(self, img):
        return transforms.Compose([
            transforms.Resize((endonet_config.img_w, endonet_config.img_h)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])(img)

    def __getitem__(self, idx: int):
        img = (
            Image.open(self.data_list[idx])
            .crop((50, 0, 804, 480))
            .convert('RGB')
        )
        return self._transform(img), self.tool_labels[idx], self.phase_labels[idx]

    def __len__(self) -> int:
        return len(self.data_list)


# ──────────────────────────────────────────────────────────────────────────
# Pool manager
# ──────────────────────────────────────────────────────────────────────────

class CholecPoolDataset:
    """
    Active learning pool manager for cholec80 training data.

    Maintains a boolean *labeled mask* over the entire training set and
    provides DataLoaders for the labeled and unlabeled subsets.

    Parameters
    ----------
    csv_path              : path to train_set_info.csv
    initial_labeled_ratio : fraction of data marked as labeled at round 0
                            (minimum 7 samples to cover all 7 phases)
    seed                  : random seed for reproducible initialisation

    Usage
    -----
    pool = CholecPoolDataset('train_set_info.csv', initial_labeled_ratio=0.05)

    # training
    labeled_loader   = pool.get_labeled_loader()
    unlabeled_loader = pool.get_unlabeled_loader()

    # after querying — pool_indices are 0-based into the *unlabeled* array
    pool.label_samples(queried_pool_indices)
    """

    def __init__(self, csv_path: str,
                 initial_labeled_ratio: float = 0.05,
                 seed: int = 42):
        self._base = CholecBaseDataset(csv_path)
        n = len(self._base)

        rng = np.random.default_rng(seed)
        n_labeled = max(7, int(n * initial_labeled_ratio))
        init_idx  = rng.choice(n, size=n_labeled, replace=False)

        self._labeled_mask = np.zeros(n, dtype=bool)
        self._labeled_mask[init_idx] = True

    # ------------------------------------------------------------------
    # Read-only properties

    @property
    def labeled_indices(self) -> np.ndarray:
        """Global indices of currently labeled samples."""
        return np.where(self._labeled_mask)[0]

    @property
    def unlabeled_indices(self) -> np.ndarray:
        """Global indices of currently unlabeled samples."""
        return np.where(~self._labeled_mask)[0]

    @property
    def n_labeled(self) -> int:
        return int(self._labeled_mask.sum())

    @property
    def n_unlabeled(self) -> int:
        return int((~self._labeled_mask).sum())

    @property
    def n_total(self) -> int:
        return len(self._base)

    # ------------------------------------------------------------------
    # Labeling

    def label_samples(self, pool_indices: np.ndarray) -> None:
        """
        Mark *pool_indices* as labeled.

        Parameters
        ----------
        pool_indices : 1-D int array
            Positions **inside the unlabeled array** (0 .. n_unlabeled-1),
            as returned by a strategy's ``query`` method.
        """
        global_idx = self.unlabeled_indices[np.asarray(pool_indices, dtype=int)]
        self._labeled_mask[global_idx] = True

    # ------------------------------------------------------------------
    # DataLoaders

    def get_labeled_loader(self, batch_size: int = 16,
                           shuffle: bool = True,
                           num_workers: int = 2) -> DataLoader:
        subset = Subset(self._base, self.labeled_indices.tolist())
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    def get_unlabeled_loader(self, batch_size: int = 64,
                             shuffle: bool = False,
                             num_workers: int = 2) -> DataLoader:
        subset = Subset(self._base, self.unlabeled_indices.tolist())
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)
