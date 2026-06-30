"""
Active learning query strategies for cholec80 EndoNet.

Seven strategies are implemented and grouped into three families:

Uncertainty sampling  (model-centric)
    RandomSamplingStrategy       – random baseline
    LeastConfidenceStrategy      – lowest max-class probability
    MarginSamplingStrategy       – smallest top-2 probability gap
    EntropySamplingStrategy      – highest predictive entropy

Bayesian approximation
    MCDropoutStrategy            – BALD via Monte Carlo Dropout

Diversity / representativeness
    CoreSetStrategy              – greedy k-center in feature space

Hybrid
    BADGEStrategy                – k-means++ on gradient embeddings

All strategies share the same interface:

    strategy.query(unlabeled_loader, n_query, labeled_loader=None)
        -> np.ndarray of shape (n_query,)
           indices into the *unlabeled* pool (0-based)

References
----------
Settles (2009) – Active Learning Literature Survey
Gal & Ghahramani (2016) – Dropout as a Bayesian Approximation
Sener & Savarese (2018) – Active Learning for Convolutional Neural Networks:
                          A Core-Set Approach  (ICLR 2018)
Ash et al. (2020) – Deep Batch Active Learning by Diverse, Uncertain
                    Gradient Lower Bounds  (ICLR 2020)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist
import torch
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base for all query strategies."""

    def __init__(self, model1, model2, device):
        """
        Parameters
        ----------
        model1 : AlexNet    – extracts features and predicts tool presence
        model2 : EasyFCNet  – predicts surgical phase from (features, tool_pred)
        device : torch.device
        """
        self.model1 = model1
        self.model2 = model2
        self.device = device

    @abstractmethod
    def query(self, unlabeled_loader, n_query: int,
              labeled_loader=None) -> np.ndarray:
        """
        Select *n_query* samples from the unlabeled pool.

        Returns
        -------
        np.ndarray of shape (n_query,) — 0-based indices into unlabeled_loader
        """

    # ------------------------------------------------------------------
    # Shared helper: deterministic forward pass

    @torch.no_grad()
    def _get_probs_and_features(self, dataloader):
        """
        Deterministic forward pass over *dataloader*.

        Returns
        -------
        probs    : np.ndarray (N, n_phases)  – softmax phase probabilities
        features : np.ndarray (N, feat_dim) – AlexNet penultimate features
        """
        self.model1.eval()
        self.model2.eval()

        all_probs, all_feats = [], []

        for X, _, _ in tqdm(dataloader, desc='  Forward pass', leave=False):
            X = X.to(self.device)
            feats, tool_logits = self.model1(X)
            combined    = torch.cat([feats, tool_logits], dim=1)
            phase_probs = torch.softmax(self.model2(combined), dim=1)

            all_probs.append(phase_probs.cpu().numpy())
            all_feats.append(feats.cpu().numpy())

        return (
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_feats, axis=0),
        )


# ──────────────────────────────────────────────────────────────────────────
# 1. Random baseline
# ──────────────────────────────────────────────────────────────────────────

class RandomSamplingStrategy(BaseStrategy):
    """
    Random uniform sampling — the standard baseline for active learning.
    No model inference required.
    """

    def query(self, unlabeled_loader, n_query: int,
              labeled_loader=None) -> np.ndarray:
        n = len(unlabeled_loader.dataset)
        return np.random.choice(n, size=min(n_query, n), replace=False)


# ──────────────────────────────────────────────────────────────────────────
# 2. Least confidence
# ──────────────────────────────────────────────────────────────────────────

class LeastConfidenceStrategy(BaseStrategy):
    """
    Select frames where the model has the *lowest* maximum class probability.

    score(x) = 1 - max_c P(y=c | x)

    A high score means the model is uncertain about its best guess.
    """

    def query(self, unlabeled_loader, n_query: int,
              labeled_loader=None) -> np.ndarray:
        probs, _ = self._get_probs_and_features(unlabeled_loader)
        scores   = 1.0 - probs.max(axis=1)      # higher = less confident
        return np.argsort(-scores)[:n_query]     # descending


# ──────────────────────────────────────────────────────────────────────────
# 3. Margin sampling
# ──────────────────────────────────────────────────────────────────────────

class MarginSamplingStrategy(BaseStrategy):
    """
    Select frames with the *smallest gap* between the top-2 class
    probabilities.

    score(x) = P(y=1st | x) - P(y=2nd | x)

    A small margin means the model is torn between two classes.
    """

    def query(self, unlabeled_loader, n_query: int,
              labeled_loader=None) -> np.ndarray:
        probs, _     = self._get_probs_and_features(unlabeled_loader)
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]   # descending
        margins      = sorted_probs[:, 0] - sorted_probs[:, 1]
        return np.argsort(margins)[:n_query]              # ascending (smallest first)


# ──────────────────────────────────────────────────────────────────────────
# 4. Entropy sampling
# ──────────────────────────────────────────────────────────────────────────

class EntropySamplingStrategy(BaseStrategy):
    """
    Select frames with the *highest* predictive entropy.

    H(y|x) = -sum_c  P(y=c|x) * log P(y=c|x)

    Entropy is maximised when the model spreads probability uniformly over
    all classes, indicating maximum uncertainty.
    """

    def query(self, unlabeled_loader, n_query: int,
              labeled_loader=None) -> np.ndarray:
        probs, _ = self._get_probs_and_features(unlabeled_loader)
        eps      = 1e-10
        entropy  = -np.sum(probs * np.log(probs + eps), axis=1)
        return np.argsort(-entropy)[:n_query]             # descending


# ──────────────────────────────────────────────────────────────────────────
# 5. MC Dropout — BALD
# ──────────────────────────────────────────────────────────────────────────

class MCDropoutStrategy(BaseStrategy):
    """
    Bayesian Active Learning by Disagreement (BALD) via Monte Carlo Dropout.

    T stochastic forward passes are made with dropout *enabled* (model in
    train mode) to approximate a posterior over model weights.

    BALD score:
        I(y ; w | x)  =  H[y|x]  -  E_w[ H[y|x,w] ]
                         --------    ------------------
                       entropy of    mean of per-pass
                       mean pred.      entropies

    A large BALD score means the model's predictions are highly variable
    across passes — i.e., both uncertain *and* not due to irreducible noise.

    Reference: Houlsby et al. (2011), Gal & Ghahramani (2016).
    """

    def __init__(self, model1, model2, device, n_forward_passes: int = 10):
        super().__init__(model1, model2, device)
        self.T = n_forward_passes

    def query(self, unlabeled_loader, n_query: int,
              labeled_loader=None) -> np.ndarray:
        mc_probs = self._mc_forward(unlabeled_loader)   # (T, N, C)
        eps      = 1e-10

        mean_probs = mc_probs.mean(axis=0)              # (N, C)

        # entropy of mean prediction
        H_mean = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)  # (N,)
        # mean of per-pass entropies
        E_H    = -np.mean(
            np.sum(mc_probs * np.log(mc_probs + eps), axis=2), axis=0)   # (N,)

        bald = H_mean - E_H
        return np.argsort(-bald)[:n_query]              # descending

    def _mc_forward(self, dataloader) -> np.ndarray:
        """
        Run T stochastic forward passes with dropout *active*.

        Returns
        -------
        np.ndarray of shape (T, N, C)
        """
        # model.train() activates dropout; torch.no_grad() skips gradients
        self.model1.train()
        self.model2.train()

        passes = []
        with torch.no_grad():
            for _ in range(self.T):
                pass_probs = []
                for X, _, _ in dataloader:
                    X  = X.to(self.device)
                    feats, tool_logits = self.model1(X)
                    combined    = torch.cat([feats, tool_logits], dim=1)
                    phase_probs = torch.softmax(self.model2(combined), dim=1)
                    pass_probs.append(phase_probs.cpu().numpy())
                passes.append(np.concatenate(pass_probs, axis=0))

        self.model1.eval()
        self.model2.eval()

        return np.stack(passes, axis=0)   # (T, N, C)


# ──────────────────────────────────────────────────────────────────────────
# 6. Core-Set (greedy k-center)
# ──────────────────────────────────────────────────────────────────────────

class CoreSetStrategy(BaseStrategy):
    """
    Core-Set selection using the greedy k-center algorithm.

    Iteratively selects the unlabeled frame that is *farthest* from all
    currently labeled frames in AlexNet feature space, minimising the
    maximum coverage radius.

    Pass *labeled_loader* to query() so that distances to existing labeled
    samples are considered.  If omitted the first selection is random.

    Reference: Sener & Savarese, ICLR 2018.
    """

    def query(self, unlabeled_loader, n_query: int,
              labeled_loader=None) -> np.ndarray:
        _, u_feats = self._get_probs_and_features(unlabeled_loader)  # (U, F)

        l_feats = None
        if labeled_loader is not None:
            _, l_feats = self._get_probs_and_features(labeled_loader)  # (L, F)

        return self._greedy_k_center(u_feats, l_feats, n_query)

    @staticmethod
    def _greedy_k_center(u_feats: np.ndarray,
                         l_feats: np.ndarray | None,
                         n_query: int) -> np.ndarray:
        """
        Greedy k-center over unlabeled feature vectors.

        At each step the point maximally distant from all selected/labeled
        points (min-distance to nearest neighbour) is chosen.
        """
        if l_feats is not None:
            # distance from each unlabeled point to its nearest labeled point
            min_dist = cdist(u_feats, l_feats, metric='euclidean').min(axis=1)
        else:
            # bootstrap with a random starting point
            first    = np.random.randint(len(u_feats))
            min_dist = cdist(u_feats,
                             u_feats[first:first + 1],
                             metric='euclidean').flatten()

        selected = []
        for _ in range(n_query):
            idx = int(np.argmax(min_dist))
            selected.append(idx)
            # update min-distances including the newly selected point
            new_dist = cdist(u_feats,
                             u_feats[idx:idx + 1],
                             metric='euclidean').flatten()
            min_dist = np.minimum(min_dist, new_dist)

        return np.array(selected)


# ──────────────────────────────────────────────────────────────────────────
# 7. BADGE
# ──────────────────────────────────────────────────────────────────────────

class BADGEStrategy(BaseStrategy):
    """
    Batch Active learning by Diverse Gradient Embeddings (BADGE).

    For each unlabeled frame, a gradient embedding is computed as the outer
    product of the loss-gradient direction and the penultimate feature vector:

        g(x) = ( p(x) - e_{y*} ) ⊗ h(x)

    where y* = argmax p(x) is the pseudo-label and h(x) is the 4103-dim
    input to the phase FC layer.

    k-means++ seeding is then applied to the embedding space to select a
    *diverse* batch that also covers high-gradient (uncertain) regions.

    Memory: the raw embedding is 7 × 4103 = 28,721-dim.  PCA is applied
    automatically when *pca_dim* > 0 to keep memory manageable.

    Reference: Ash et al., ICLR 2020.
    """

    def __init__(self, model1, model2, device, pca_dim: int = 256):
        super().__init__(model1, model2, device)
        self.pca_dim = pca_dim

    def query(self, unlabeled_loader, n_query: int,
              labeled_loader=None) -> np.ndarray:
        embeddings = self._gradient_embeddings(unlabeled_loader)  # (N, D)

        # optional PCA to reduce memory before k-means++
        if (self.pca_dim > 0
                and embeddings.shape[1] > self.pca_dim
                and embeddings.shape[0] > self.pca_dim):
            from sklearn.decomposition import PCA
            embeddings = PCA(n_components=self.pca_dim,
                             random_state=0).fit_transform(embeddings)

        return self._kmeans_pp(embeddings, n_query)

    # ------------------------------------------------------------------

    def _gradient_embeddings(self, dataloader) -> np.ndarray:
        """Compute per-sample gradient embeddings (N, n_classes * feat_dim)."""
        self.model1.eval()
        self.model2.eval()

        all_emb = []

        with torch.no_grad():
            for X, _, _ in tqdm(dataloader, desc='  BADGE embeddings', leave=False):
                X  = X.to(self.device)
                feats, tool_logits = self.model1(X)
                combined = torch.cat([feats, tool_logits], dim=1)   # (B, 4096+7=4103)
                phase_log = self.model2(combined)
                probs     = torch.softmax(phase_log, dim=1)           # (B, C)
                pred      = probs.argmax(dim=1)                       # (B,)

                probs_np = probs.cpu().numpy()       # (B, C)
                h_np     = combined.cpu().numpy()    # (B, F)
                C        = probs_np.shape[1]
                B        = X.size(0)

                batch_emb = np.empty((B, C * h_np.shape[1]), dtype=np.float32)
                for b in range(B):
                    one_hot      = np.zeros(C, dtype=np.float32)
                    one_hot[int(pred[b])] = 1.0
                    grad         = probs_np[b] - one_hot              # (C,)
                    batch_emb[b] = np.outer(grad, h_np[b]).ravel()   # (C*F,)

                all_emb.append(batch_emb)

        return np.concatenate(all_emb, axis=0)   # (N, C*F)

    @staticmethod
    def _kmeans_pp(embeddings: np.ndarray, n_query: int) -> np.ndarray:
        """
        k-means++ seeding: select *n_query* diverse samples.

        Samples are drawn proportionally to their squared distance to the
        nearest already-selected centre (L2-normalised embeddings).
        """
        n = len(embeddings)
        # L2-normalise so that dot-product and Euclidean distance are related
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        emb_n = embeddings / norms

        selected: list[int] = []
        first = np.random.randint(n)
        selected.append(first)
        min_sq_dist = np.sum((emb_n - emb_n[first]) ** 2, axis=1)

        for _ in range(n_query - 1):
            probs = min_sq_dist / (min_sq_dist.sum() + 1e-10)
            idx   = int(np.random.choice(n, p=probs))
            selected.append(idx)
            new_sq      = np.sum((emb_n - emb_n[idx]) ** 2, axis=1)
            min_sq_dist = np.minimum(min_sq_dist, new_sq)

        return np.array(selected)


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────

_STRATEGY_MAP: dict[str, type[BaseStrategy]] = {
    'random':           RandomSamplingStrategy,
    'least_confidence': LeastConfidenceStrategy,
    'margin':           MarginSamplingStrategy,
    'entropy':          EntropySamplingStrategy,
    'mc_dropout':       MCDropoutStrategy,
    'coreset':          CoreSetStrategy,
    'badge':            BADGEStrategy,
}


def get_strategy(name: str, model1, model2, device, **kwargs) -> BaseStrategy:
    """
    Instantiate a strategy by name.

    Parameters
    ----------
    name     : one of the keys in _STRATEGY_MAP
    model1   : AlexNet
    model2   : EasyFCNet
    device   : torch.device
    **kwargs : forwarded to the strategy constructor
               (e.g. n_forward_passes=10 for mc_dropout,
                     pca_dim=128        for badge)

    Raises
    ------
    ValueError if *name* is not recognised.
    """
    if name not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {sorted(_STRATEGY_MAP.keys())}"
        )
    return _STRATEGY_MAP[name](model1, model2, device, **kwargs)
