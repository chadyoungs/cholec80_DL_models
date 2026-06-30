"""
Active learning configuration for cholec80 EndoNet.

All AL-specific hyper-parameters are defined here so that they can be
adjusted without touching the main training script.
"""

# ──────────────────────────────────────────────────────────────────────────
# Default AL loop settings  (overridable via CLI in active_train.py)
# ──────────────────────────────────────────────────────────────────────────
al_params = {
    # fraction of the training pool to mark as labeled at round 0
    'initial_labeled_ratio': 0.05,
    # number of frames to annotate per round
    'query_budget': 200,
    # total active-learning cycles
    'n_rounds': 10,
    # gradient-descent epochs inside each AL round
    'epochs_per_round': 20,
    # default query strategy
    'strategy': 'entropy',
}

# ──────────────────────────────────────────────────────────────────────────
# Model training settings  (mirrors EndoNet/config.py where needed)
# ──────────────────────────────────────────────────────────────────────────
training_params = {
    'batch_size':               16,
    'learning_rate_feature':    1e-3,
    'learning_rate_classifier': 1e-2,
    'momentum':                 0.9,
    # sigmoid threshold for multi-label tool detection
    'threshold':                0.5,
}

# ──────────────────────────────────────────────────────────────────────────
# Strategy-specific settings
# ──────────────────────────────────────────────────────────────────────────

# Number of stochastic forward passes for MC Dropout (BALD)
mc_dropout_passes = 10

# Maximum embedding dimensionality for BADGE before applying PCA.
# The full gradient embedding is (n_phases x feat_dim) = 7 x 4103 = 28721.
# PCA keeps memory manageable for large unlabeled pools.
badge_pca_dim = 256

# ──────────────────────────────────────────────────────────────────────────
# Available strategies  (for reference / validation)
# ──────────────────────────────────────────────────────────────────────────
STRATEGIES = {
    'random':           'Random baseline — uniform sampling from the unlabeled pool',
    'least_confidence': 'Uncertainty — select frames with the lowest max-class probability',
    'margin':           'Uncertainty — select frames with the smallest top-2 probability gap',
    'entropy':          'Uncertainty — select frames with the highest predictive entropy',
    'mc_dropout':       'Bayesian   — BALD via Monte Carlo Dropout',
    'coreset':          'Diversity  — greedy k-center in AlexNet feature space (Core-Set)',
    'badge':            'Hybrid     — k-means++ on gradient embeddings (BADGE)',
}
