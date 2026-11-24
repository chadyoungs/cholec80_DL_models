"""Query strategies operating on *cached* model outputs.

All functions take arrays already produced by a single forward pass over the
unlabeled pool, which keeps every round to one (or, for MC Dropout, ``T``)
inference passes regardless of how many strategies are compared.

Every strategy returns indices into the unlabeled pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CandidateSet:
    probs: np.ndarray            # (N, C) softmax phase probabilities
    feats: np.ndarray            # (N, D) penultimate features


STRATEGIES: dict[str, str] = {
    "random": "Uniform sampling baseline",
    "least_confidence": "Lowest max class probability",
    "margin": "Smallest top-2 probability gap",
    "entropy": "Highest predictive entropy",
    "mc_dropout": "BALD via Monte-Carlo dropout",
    "coreset": "Greedy k-center in feature space",
    "badge": "k-means++ on gradient embeddings (random-projected)",
}


# --------------------------------------------------------------------------- #
# Uncertainty family
# --------------------------------------------------------------------------- #

def _entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def least_confidence(cand: CandidateSet, n: int) -> np.ndarray:
    return np.argsort(-(1.0 - cand.probs.max(axis=1)))[:n]


def margin(cand: CandidateSet, n: int) -> np.ndarray:
    ordered = np.sort(cand.probs, axis=1)
    gaps = ordered[:, -1] - ordered[:, -2] if ordered.shape[1] > 1 else ordered[:, -1]
    return np.argsort(gaps)[:n]


def entropy(cand: CandidateSet, n: int) -> np.ndarray:
    return np.argsort(-_entropy(cand.probs))[:n]


def bald(mc_probs: np.ndarray, n: int) -> np.ndarray:
    """BALD score ``H[mean p] - mean_t H[p_t]`` for ``mc_probs`` of shape (T,N,C)."""
    mean_probs = mc_probs.mean(axis=0)
    h_mean = _entropy(mean_probs)
    h_each = -np.mean(
        np.sum(mc_probs * np.log(mc_probs + 1e-10), axis=2), axis=0
    )
    return np.argsort(-(h_mean - h_each))[:n]


# --------------------------------------------------------------------------- #
# Diversity family
# --------------------------------------------------------------------------- #

def _min_sq_dist(points: np.ndarray, centers: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out = np.empty(len(points), dtype=np.float32)
    for i in range(0, len(points), chunk):
        block = points[i : i + chunk]
        d = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out[i : i + chunk] = d.min(axis=1)
    return out


def coreset(
    cand: CandidateSet,
    n: int,
    labeled_feats: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greedy k-center: iteratively pick the point farthest from the selection."""
    rng = rng or np.random.default_rng()
    feats = cand.feats.astype(np.float32, copy=False)

    if labeled_feats is not None and len(labeled_feats):
        min_sq = _min_sq_dist(feats, labeled_feats.astype(np.float32, copy=False))
    else:
        seed = int(rng.integers(len(feats)))
        min_sq = _min_sq_dist(feats, feats[[seed]])

    selected: list[int] = []
    for _ in range(min(n, len(feats))):
        idx = int(np.argmax(min_sq))
        selected.append(idx)
        min_sq = np.minimum(min_sq, _min_sq_dist(feats, feats[[idx]]))
        min_sq[idx] = -np.inf  # never select the same point twice
    return np.array(selected, dtype=np.int64)


def _random_projection(dim: int, proj_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, proj_dim)).astype(np.float32)


def badge(
    cand: CandidateSet,
    n: int,
    proj_dim: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """BADGE (Ash et al., ICLR 2020) with a random projection for tractability.

    The exact gradient embedding ``(p - onehot(y*)) ⊗ h`` has
    ``C * D`` dimensions (7 * 2048 = 14336 for ResNet-50), so features are
    randomly projected to ``proj_dim`` before k-means++ seeding. This is a
    documented approximation, not the exact paper formulation.
    """
    rng = rng or np.random.default_rng()
    probs = cand.probs.astype(np.float32, copy=False)
    feats = cand.feats.astype(np.float32, copy=False)

    if feats.shape[1] > proj_dim:
        feats = feats @ _random_projection(feats.shape[1], proj_dim, rng)

    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), probs.argmax(axis=1)] = 1.0
    grad = probs - onehot                                    # (N,C)
    emb = (grad[:, :, None] * feats[:, None, :]).reshape(len(probs), -1)  # (N, C*proj)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= np.maximum(norms, 1e-10)

    selected: list[int] = []
    first = int(rng.integers(len(emb)))
    selected.append(first)
    min_sq = ((emb - emb[first]) ** 2).sum(axis=1)

    for _ in range(1, min(n, len(emb))):
        p = min_sq / (min_sq.sum() + 1e-10)
        idx = int(rng.choice(len(emb), p=p))
        selected.append(idx)
        new = ((emb - emb[idx]) ** 2).sum(axis=1)
        min_sq = np.minimum(min_sq, new)
        min_sq[selected] = 0.0        # never re-select, but keep probabilities non-negative
    return np.array(selected, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def query_indices(
    name: str,
    cand: CandidateSet,
    n: int,
    *,
    labeled_feats: np.ndarray | None = None,
    mc_probs: np.ndarray | None = None,
    proj_dim: int = 64,
    seed: int | None = None,
) -> np.ndarray:
    """Return ``n`` indices into the unlabeled pool for strategy ``name``."""
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}")

    n = int(min(n, len(cand.probs)))
    if n <= 0:
        return np.zeros(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    if name == "random":
        return rng.choice(len(cand.probs), size=n, replace=False)
    if name == "least_confidence":
        return least_confidence(cand, n)
    if name == "margin":
        return margin(cand, n)
    if name == "entropy":
        return entropy(cand, n)
    if name == "mc_dropout":
        if mc_probs is None:
            raise ValueError("mc_dropout requires mc_probs (T, N, C)")
        return bald(mc_probs, n)
    if name == "coreset":
        return coreset(cand, n, labeled_feats=labeled_feats, rng=rng)
    if name == "badge":
        return badge(cand, n, proj_dim=proj_dim, rng=rng)
    raise AssertionError("unreachable")
