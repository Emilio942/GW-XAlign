"""Baselines for zero-shot alignment (all operate on the same cached features)."""
from __future__ import annotations

import numpy as np

from gwxalign.core import unit


def _l2(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def baseline_relative_representations(ZA, ZB, yB, yA, n_anchors=256, k=10, seed=0):
    """Moschella et al. (2023): represent samples via similarities to anchors,
    kNN-classify in the shared relative space."""
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(seed)
    anchors = ZB[rng.choice(len(ZB), min(n_anchors, len(ZB)), replace=False)]
    a_n, b_n, an_n = _l2(ZA), _l2(ZB), _l2(anchors)
    RA = a_n @ an_n.T          # (nA, anchors)
    RB = b_n @ an_n.T          # (nB, anchors)
    knn = KNeighborsClassifier(n_neighbors=k).fit(RB, yB)
    return float(knn.score(RA, yA))


def baseline_naive_ot(ZA, ZB, yB, yA, k=10, eps=0.02):
    """Standard entropic Wasserstein on euclidean distances (ignores GW structure)."""
    import ot
    from scipy.spatial.distance import cdist
    from sklearn.neighbors import KNeighborsClassifier
    C = cdist(ZA, ZB)
    C = C / (C.max() + 1e-9)
    T = ot.sinkhorn(unit(len(ZA)), unit(len(ZB)), C, reg=eps)
    B = (T / unit(len(ZA))[:, None]) @ ZB
    knn = KNeighborsClassifier(n_neighbors=k).fit(ZB, yB)
    return float(knn.score(B, yA))


def baseline_random_map(ZA, ZB, yB, yA, k=10, seed=0):
    """Lower bound: random semi-orthogonal projection of ZA into dim(ZB)."""
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(seed)
    dA, dB = ZA.shape[1], ZB.shape[1]
    Q, _ = np.linalg.qr(rng.normal(size=(dA, dB)))
    B = (ZA - ZA.mean(0)) @ Q + ZB.mean(0)
    knn = KNeighborsClassifier(n_neighbors=k).fit(ZB, yB)
    return float(knn.score(B, yA))
