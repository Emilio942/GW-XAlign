"""Baselines for zero-shot alignment (all operate on the same cached features)."""
from __future__ import annotations

import numpy as np

from gwxalign.core import unit


def _l2(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def _pad_to(Z, d):
    """Zero-pad features to dimension d (cosine-invariant)."""
    if Z.shape[1] >= d:
        return Z
    return np.hstack([Z, np.zeros((len(Z), d - Z.shape[1]), dtype=Z.dtype)])


def baseline_relative_representations(ZA, ZB, yB, yA, n_anchors=256, k=10, seed=0):
    """Moschella et al. (2023): represent samples via similarities to anchors,
    kNN-classify in the shared relative space. Dimension mismatch is handled
    by zero-padding source and anchors to a common dimension (cosine-invariant)."""
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(seed)
    anchors = ZB[rng.choice(len(ZB), min(n_anchors, len(ZB)), replace=False)]
    d = max(ZA.shape[1], ZB.shape[1])
    a_n, b_n, an_n = _l2(_pad_to(ZA, d)), _l2(ZB), _l2(_pad_to(anchors, d))
    RA = a_n @ an_n.T          # (nA, anchors)
    RB = b_n @ an_n.T          # (nB, anchors)
    knn = KNeighborsClassifier(n_neighbors=k).fit(RB, yB)
    return float(knn.score(RA, yA))


def baseline_naive_ot(ZA, ZB, yB, yA, k=10, eps=0.02):
    """Standard entropic Wasserstein on euclidean distances (ignores GW structure).

    NOTE: under dimension mismatch (dA != dB) euclidean geometry between the
    two clouds is undefined; we zero-pad the smaller side, which makes the
    cost degenerate (no cross term). The resulting baseline is therefore
    expected to be weak exactly in the mismatch regime — that gap is what
    GW alignment fills (RQ3)."""
    import ot
    from sklearn.neighbors import KNeighborsClassifier
    d = max(ZA.shape[1], ZB.shape[1])
    XA, XB = _pad_to(ZA, d), _pad_to(ZB, d)
    d2 = (np.sum(XA**2, axis=1)[:, None] + np.sum(XB**2, axis=1)[None, :]
          - 2.0 * (XA @ XB.T))
    C = np.sqrt(np.maximum(d2, 0.0))
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
    if dA >= dB:
        Q, _ = np.linalg.qr(rng.normal(size=(dA, dB)))       # (dA, dB) orthonormal cols
        B = (ZA - ZA.mean(0)) @ Q
    else:
        Q, _ = np.linalg.qr(rng.normal(size=(dB, dA)))       # (dB, dA) orthonormal cols
        B = (ZA - ZA.mean(0)) @ Q.T                          # (dA, dB) orthonormal rows
    B = B + ZB.mean(0)
    knn = KNeighborsClassifier(n_neighbors=k).fit(ZB, yB)
    return float(knn.score(B, yA))
