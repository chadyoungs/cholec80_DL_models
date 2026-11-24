"""Query strategies return valid, unique indices into the unlabeled pool."""

from __future__ import annotations

import numpy as np
import pytest

from cholec80.active_learning import STRATEGIES, CandidateSet, query_indices


@pytest.fixture()
def candidates():
    rng = np.random.default_rng(0)
    n, c, d = 60, 7, 16
    logits = rng.normal(size=(n, c))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    feats = rng.normal(size=(n, d)).astype(np.float32)
    return CandidateSet(probs=probs, feats=feats)


@pytest.mark.parametrize("name", [s for s in STRATEGIES if s != "mc_dropout"])
def test_strategy_shapes(name, candidates):
    n_query = 10
    selected = query_indices(name, candidates, n_query, seed=0)
    assert selected.shape == (n_query,)
    assert len(set(selected.tolist())) == n_query
    assert selected.min() >= 0 and selected.max() < len(candidates.probs)


def test_mc_dropout_requires_passes(candidates):
    with pytest.raises(ValueError):
        query_indices("mc_dropout", candidates, 5)


def test_mc_dropout_bald(candidates):
    rng = np.random.default_rng(1)
    mc = rng.dirichlet(np.ones(7), size=(5, len(candidates.probs)))  # (T,N,C)
    selected = query_indices("mc_dropout", candidates, 6, mc_probs=mc)
    assert len(set(selected.tolist())) == 6


def test_coreset_uses_labeled_features(candidates):
    labeled = candidates.feats[:5] + 100.0   # far away -> selection must prefer unlabeled points
    selected = query_indices("coreset", candidates, 4, labeled_feats=labeled, seed=2)
    assert len(set(selected.tolist())) == 4


def test_margin_and_entropy_prefer_uncertain(candidates):
    uniform = np.full((3, 7), 1 / 7)
    confident = np.tile(np.array([1.0, 0, 0, 0, 0, 0, 0]), (3, 1))
    cand = CandidateSet(probs=np.vstack([confident, uniform]), feats=candidates.feats[:6])
    for name in ("margin", "entropy", "least_confidence"):
        top = set(query_indices(name, cand, 3, seed=0).tolist())
        assert top == {3, 4, 5}


def test_query_caps_at_pool_size(candidates):
    selected = query_indices("random", candidates, 1000)
    assert len(selected) == len(candidates.probs)
