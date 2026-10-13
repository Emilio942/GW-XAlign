"""GW-XAlign core library: entropic Gromov-Wasserstein solvers, barycentric
pushforward maps, spectral diagnostics, and evaluation metrics.

Research library for the GW-XAlign project:
  Zero-Shot Cross-Architecture Representation Alignment via
  Entropic Gromov-Wasserstein Optimal Transport.
"""
from __future__ import annotations

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Metric-measure space construction
# ---------------------------------------------------------------------------

def cosine_distance_matrix(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalized cosine distance matrix d(z_i, z_j) = 1 - cos(z_i, z_j) in [0, 2]."""
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + eps)
    D = 1.0 - Zn @ Zn.T
    return np.clip(D, 0.0, 2.0)


def euclidean_distance_matrix(Z: np.ndarray) -> np.ndarray:
    sq = np.sum(Z * Z, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    return np.sqrt(np.maximum(D2, 0.0))


def normalize_diameter(D: np.ndarray) -> np.ndarray:
    """Scale distance matrix to unit diameter (numerical stability for EGW)."""
    m = D.max()
    return D / m if m > 0 else D


def knn_geodesic_distance(Z: np.ndarray, k: int = 10) -> np.ndarray:
    """Intrinsic geodesic distance over the symmetrized k-NN graph (Floyd-Warshall)."""
    n = Z.shape[0]
    D_eucl = euclidean_distance_matrix(Z)
    W = np.full((n, n), np.inf)
    np.fill_diagonal(W, 0.0)
    # directed kNN edges, symmetrized
    idx = np.argsort(D_eucl, axis=1)[:, 1 : k + 1]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    W[rows, cols] = D_eucl[rows, cols]
    W = np.minimum(W, W.T)
    # Floyd-Warshall
    for kk in range(n):
        W = np.minimum(W, W[:, kk, None] + W[None, kk, :])
    finite = np.isfinite(W)
    return np.where(finite, W, D_eucl * 2.0)


# ---------------------------------------------------------------------------
# EGW solvers
# ---------------------------------------------------------------------------

def gw_value(C1, C2, T, p=None, q=None):
    """Quadratic GW energy sum L(d1_ik, d2_jl) T_ij T_kl, L(u,v)=1/2(u-v)^2."""
    n, m = C1.shape[0], C2.shape[0]
    if p is None:
        p = unit(n)
    if q is None:
        q = unit(m)
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun="square_loss")
    return float(ot.gromov.gwloss(constC, hC1, hC2, T))


def solve_egw(
    C1: np.ndarray,
    C2: np.ndarray,
    p: np.ndarray = None,
    q: np.ndarray = None,
    eps: float = 0.01,
    num_iter: int = 1000,
    tol: float = 1e-9,
    solver: str = "PPA",
    verbose: bool = False,
) -> tuple:
    """Entropic GW with quadratic loss. Returns (T, log) with T the coupling."""
    n, m = C1.shape[0], C2.shape[0]
    if p is None:
        p = unit(n)
    if q is None:
        q = unit(m)
    T, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q,
        loss_fun="square_loss",
        epsilon=eps,
        solver=solver,
        max_iter=num_iter,
        tol=tol,
        verbose=verbose,
        log=True,
        numItermax=50,
        stopThr=1e-8,
    )
    T = np.asarray(T)
    log = dict(log) if log else {}
    log["loss"] = gw_value(C1, C2, T, p, q)
    return T, log


def solve_gw(C1, C2, p=None, q=None, num_iter: int = 500):
    """Unregularized GW (condition-gradient)."""
    n, m = C1.shape[0], C2.shape[0]
    if p is None:
        p = unit(n)
    if q is None:
        q = unit(m)
    T, log = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, loss_fun="square_loss",
        max_iter=num_iter, tol_rel=1e-9, tol_abs=1e-12, verbose=False, log=True,
    )
    T = np.asarray(T)
    log = dict(log) if log else {}
    log["loss"] = gw_value(C1, C2, T, p, q)
    return T, log


# ---------------------------------------------------------------------------
# Barycentric pushforward (in-sample and out-of-sample)
# ---------------------------------------------------------------------------

def barycentric_map(T: np.ndarray, Y: np.ndarray, p: np.ndarray = None) -> np.ndarray:
    """In-sample barycentric projection: x_i -> sum_j T_ij y_j / p_i."""
    n = T.shape[0]
    if p is None:
        p = np.full(n, 1.0 / n)
    return (T / p[:, None]) @ Y


def out_of_sample_map(
    x: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    p: np.ndarray = None,
    kernel_width: float = None,
    k: int = 10,
) -> np.ndarray:
    """Zero-shot barycentric pushforward T_gamma(x) via localized KDE weights.

    w_i = kappa(x, x_i) (Gaussian kernel over X) with adaptive bandwidth given
    by the distance to the k-th nearest source point; then
    y* = argmin_y sum_j w'_j d2(y, y_j)^2 with w'_j = sum_i (T_ij/p_i) w_i.
    For Euclidean d2 the minimizer is the weighted mean.
    """
    n = X.shape[0]
    if p is None:
        p = unit(n)
    d2 = np.sum((X - x) ** 2, axis=1)
    if kernel_width is None:
        # adaptive local bandwidth: distance to the k-th nearest neighbor
        knn_d = np.sqrt(np.sort(d2)[min(k, n - 1)])
        kernel_width = max(knn_d, 1e-9)
    w = np.exp(-d2 / (2 * kernel_width**2))
    wp = (T / p[:, None]).T @ w          # shape (m,)
    s = wp.sum()
    if s <= 0 or not np.isfinite(s):
        j = int(np.argmax(wp))
        return Y[j]
    return (wp @ Y) / s


def solve_egw_gpu(
    C1: np.ndarray,
    C2: np.ndarray,
    p: np.ndarray = None,
    q: np.ndarray = None,
    eps: float = 0.02,
    max_iter: int = 200,
    tol: float = 1e-8,
    device: str = "cuda",
) -> tuple:
    """GPU (torch float32) PPA solver; falls back to CPU on failure.

    Returns (T float64 numpy, log with 'loss', 'converged', 'last_err').
    """
    try:
        import torch
        import ot as _ot
        n, m = C1.shape[0], C2.shape[0]
        if p is None:
            p = unit(n)
        if q is None:
            q = unit(m)
        dev = device if torch.cuda.is_available() else "cpu"
        # float32 underflows for small eps: exp(-C/eps) -> 0 (NaN in log-domain).
        dtype = torch.float32 if (eps >= 0.01 and dev == "cuda") else torch.float64
        pt = torch.as_tensor(p, dtype=dtype, device=dev)
        qt = torch.as_tensor(q, dtype=dtype, device=dev)
        C1t = torch.as_tensor(C1, dtype=dtype, device=dev)
        C2t = torch.as_tensor(C2, dtype=dtype, device=dev)
        T, log = _ot.gromov.entropic_gromov_wasserstein(
            C1t, C2t, pt, qt, loss_fun="square_loss", epsilon=eps,
            solver="PPA", max_iter=max_iter, tol=tol, log=True,
            numItermax=50, stopThr=1e-8)
        T = np.asarray(T.detach().cpu().numpy(), dtype=np.float64)
        if not np.isfinite(T).all() or not np.isfinite(T.sum()) or T.sum() <= 0:
            raise FloatingPointError("non-finite coupling (eps too small for dtype)")
        T = np.clip(T, 0.0, None)
        T /= T.sum()
        err = float(log["err"][-1]) if log and log.get("err") else float("nan")
        return T, dict(loss=gw_value(C1, C2, T, p, q), last_err=err,
                       converged=bool(err < 10 * tol))
    except Exception as e:  # pragma: no cover
        T, log = solve_egw(C1, C2, p, q, eps=eps, num_iter=max_iter, tol=tol)
        log["converged"] = bool(log.get("err", [1])[-1] < 10 * tol) if log.get("err") else False
        return T, log


def neighborhood_precision(T, Y, gt, p=None, k: int = 10) -> float:
    """Fraction of i whose ground-truth partner gt[i] is among the k nearest
    target points of the barycentric image of x_i. Robust to local density."""
    from scipy.spatial.distance import cdist
    n = T.shape[0]
    if p is None:
        p = unit(n)
    B = (T / p[:, None]) @ Y
    nn = np.argsort(cdist(B, Y), axis=1)[:, :k]
    return float(np.mean([gt[i] in nn[i] for i in range(n)]))


# ---------------------------------------------------------------------------
# Spectral diagnostics (Research Question 2)
# ---------------------------------------------------------------------------

def spectrum(D: np.ndarray, k_max: int = None) -> np.ndarray:
    """Sorted eigenvalues (descending) of a symmetric distance matrix."""
    lam = np.linalg.eigvalsh(D)[::-1]
    return lam[:k_max] if k_max else lam


def spectral_distance(D1: np.ndarray, D2: np.ndarray, k: int = None,
                      normalized: bool = True) -> float:
    """sum_i |lambda_i(D1) - lambda_i(D2)| over aligned top-k eigenvalues.

    Both matrices are diameter-normalized first if `normalized`, making the
    statistic scale-invariant (raw spectra are not).
    """
    if normalized:
        D1, D2 = normalize_diameter(D1), normalize_diameter(D2)
    l1, l2 = spectrum(D1), spectrum(D2)
    k = min(len(l1), len(l2)) if k is None else min(k, len(l1), len(l2))
    return float(np.sum(np.abs(l1[:k] - l2[:k])))


def spectral_distance_wasserstein(D1: np.ndarray, D2: np.ndarray,
                                  normalized: bool = True) -> float:
    """W1 distance between empirical eigenvalue distributions (scale-normalized)."""
    if normalized:
        D1, D2 = normalize_diameter(D1), normalize_diameter(D2)
    l1, l2 = spectrum(D1), spectrum(D2)
    allv = np.sort(np.concatenate([l1, l2]))
    c1 = np.searchsorted(np.sort(l1), allv, side="right") / len(l1)
    c2 = np.searchsorted(np.sort(l2), allv, side="right") / len(l2)
    return float(np.trapezoid(np.abs(c1 - c2), allv))


# ---------------------------------------------------------------------------
# Correspondence / alignment evaluation
# ---------------------------------------------------------------------------

def matching_accuracy(T: np.ndarray, ground_truth: np.ndarray) -> float:
    """Fraction of recovered pairs under a known bijection ground_truth[i]=j.

    Uses the coupling: for each i take argmax_j T_ij (biased toward marginals),
    plus Hungarian matching for comparison.
    """
    n = T.shape[0]
    pred = np.argmax(T, axis=1)
    acc_argmax = float(np.mean(pred == ground_truth))

    # Hungarian on -T (maximize total mass of matched pairs)
    rows, cols = linear_sum_assignment(-T)
    acc_hung = float(np.mean(cols == ground_truth))
    return (acc_argmax + acc_hung) / 2.0


def knn_transfer_accuracy(Z_src_mapped: np.ndarray, Z_tgt: np.ndarray,
                          y_tgt: np.ndarray, y_src: np.ndarray,
                          k: int = 10) -> float:
    """Accuracy of predicting y_src labels via kNN vote in the mapped space.

    Measures whether mapped source activations land near target activations
    of the same semantic class.
    """
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(Z_tgt, y_tgt)
    return float(knn.score(Z_src_mapped, y_src))


def knn_transfer_accuracy_src(Z_src: np.ndarray, y_src: np.ndarray,
                              Z_tgt: np.ndarray, y_tgt: np.ndarray,
                              k: int = 10) -> float:
    """Symmetric direction: classify target points by kNN among source points."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(Z_src, y_src)
    return float(knn.score(Z_tgt, y_tgt))


def relative_task_loss(loss_aligned: float, loss_target: float) -> float:
    return loss_aligned - loss_target


# ---------------------------------------------------------------------------
# Synthetic architecture pair generators (controlled ground truth)
# ---------------------------------------------------------------------------

def make_aligned_pair(
    n: int,
    d1: int,
    d2: int,
    seed: int = 0,
    distortion: float = 0.0,
    noise: float = 0.0,
    nonlinear: bool = True,
) -> tuple:
    """Create a source cloud Z1 (R^d1) and a target cloud Z2 (R^d2) that is a
    distorted observation of the same latent structure (swiss roll manifold,
    each point has a unique geometric fingerprint => well-posed bijection).

    distortion >= 0: strength of a smooth random nonlinear warp of Z2.
    noise >= 0: additive isotropic noise on Z2.

    Returns (Z1, Z2, gt): gt[i] = row index of the Z2 point corresponding to Z1[i].
    """
    rng = np.random.default_rng(seed)
    # latent swiss roll in R^3
    t = rng.uniform(0.0, 3.0 * np.pi, size=n)
    h = rng.uniform(0.0, 10.0, size=n)
    L = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
    L = (L - L.mean(0)) / L.std(0)

    def embed(X, d):
        if d >= 3:
            Q, _ = np.linalg.qr(rng.normal(size=(d, 3)))   # Q: (d,3) orthonormal cols
            return X @ Q.T * np.sqrt(d / 3.0)
        raise ValueError("need d>=3")

    Z1 = embed(L, d1)

    # architecture-2 observation: projection, per-axis gain, monotone map
    P = rng.normal(size=(d1, d2)) / np.sqrt(d1)
    Z2 = Z1 @ P
    gain = 1.0 + 0.25 * rng.normal(size=d2)
    Z2 = Z2 * gain[None, :]
    if nonlinear:
        Z2 = np.tanh(Z2)
    if distortion > 0:
        # smooth random warp: Z2 += distortion * scale * tanh(Z2 A) B
        A = rng.normal(size=(d2, max(2, d2 // 2))) / np.sqrt(d2)
        B = rng.normal(size=(max(2, d2 // 2), d2)) / np.sqrt(max(2, d2 // 2))
        scale = np.median(np.abs(Z2)) + 1e-9
        Z2 = Z2 + distortion * scale * np.tanh(Z2 @ A) @ B
    if noise > 0:
        Z2 = Z2 + noise * (np.abs(Z2).mean() + 1e-9) * rng.normal(size=Z2.shape)

    perm = rng.permutation(n)
    Z2 = Z2[perm]
    gt = np.empty(n, dtype=int)
    gt[perm] = np.arange(n)   # Z2[gt[i]] corresponds to Z1[i]
    return Z1, Z2, gt


def make_fragmented_moe_pair(
    n: int,
    d: int,
    n_experts: int = 8,
    gap: float = 5.0,
    seed: int = 0,
) -> tuple:
    """Simulate MoE routing fragmentation: n_experts well-separated clusters;
    within-expert structure is a smooth manifold, across experts distances are
    'topological noise' (large gaps unrelated to the target geometry)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_experts, d)) * gap
    assign = rng.integers(0, n_experts, size=n)
    Z = np.zeros((n, d))
    for i in range(n):
        Z[i] = centers[assign[i]] + 0.5 * rng.normal(size=d)
    return Z, assign


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def unit(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
