"""Exp J: Deleted-edge corruption model for MoE fragmentation + repairs.

The affine-inflation model (Exp F/G) preserves ordinal inter-expert structure.
A harsher, arguably more realistic MoE artifact is *uninformative* inter-expert
distances (deleted edges): inter entries ~ i.i.d. noise. Repairs tested:

  none      : corrupt D as-is
  deoffset  : quantile de-offset (worked for inflation; expected to fail here)
  mds       : impute inter-expert entries from an MDS embedding of the
              K x K aggregate expert-distance matrix + median intra offset
              (uses router-known expert labels only)

Metrics: prec@1/@10 vs each seed's clean baseline.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, make_aligned_pair,
                           neighborhood_precision, normalize_diameter,
                           solve_egw_gpu, unit)
from exp_f_moe import fragment


def corrupt_deleted(D, assign, rng):
    """Replace inter-expert distances by i.i.d. uniform noise in [q10, q95]
    of the intra-expert distribution (ordinal information destroyed)."""
    n = len(assign)
    same = assign[:, None] == assign[None, :]
    intra = D[same & ~np.eye(n, dtype=bool)]
    lo, hi = np.quantile(intra, 0.10), np.quantile(intra, 0.95)
    Dc = D.copy()
    noise = rng.uniform(lo, hi, size=D.shape)
    Dc = np.where(same, D, (noise + noise.T) / 2.0)
    return normalize_diameter(Dc)


def mds_impute(D, assign):
    """Impute inter-expert entries from a 2D classical-MDS embedding of the
    K x K median expert-distance matrix; inter(i,j) := ||c_a - c_b|| + med_intra."""
    K = assign.max() + 1
    n = len(assign)
    same = assign[:, None] == assign[None, :]
    intra_med = np.median(D[same & ~np.eye(n, dtype=bool)])
    # aggregate expert-level distance matrix
    A = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            if a != b:
                m = (assign[:, None] == a) & (assign[None, :] == b)
                A[a, b] = np.median(D[m])
    A = (A + A.T) / 2.0
    # classical MDS
    J = np.eye(K) - np.ones((K, K)) / K
    Bm = -0.5 * J @ A @ J
    w, V = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:2]
    X = V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))
    Dc = D.copy()
    for i in range(n):
        for j in range(n):
            if not same[i, j]:
                Dc[i, j] = np.linalg.norm(X[assign[i]] - X[assign[j]]) + intra_med
    return normalize_diameter(Dc)


def main():
    t0 = time.time()
    N, K, EPS = 400, 8, 0.03
    rows = []
    for seed in [0, 1, 2]:
        Z1, Z2, gt = make_aligned_pair(n=N, d1=12, d2=7, seed=100 + seed,
                                       distortion=0.3, noise=0.05)
        D2 = normalize_diameter(euclidean_distance_matrix(Z2))
        ang = np.arctan2(Z1[:, 2], Z1[:, 0])
        assign = np.unique(
            np.clip(np.digitize(ang, np.linspace(-np.pi, np.pi, K + 1)),
                    0, K - 1), return_inverse=True)[1]
        K_eff = assign.max() + 1
        D1 = normalize_diameter(euclidean_distance_matrix(Z1))
        # clean baseline
        T, _ = solve_egw_gpu(D1, D2, eps=EPS, max_iter=250, tol=1e-8)
        b1 = neighborhood_precision(T, Z2, gt, unit(N), 1)
        b10 = neighborhood_precision(T, Z2, gt, unit(N), 10)
        rows.append(dict(seed=seed, corr="clean", prec1=b1, prec10=b10, rel10=1.0))
        print(f"[J] seed={seed} CLEAN prec1={b1:.3f} prec10={b10:.3f}", flush=True)
        rng = np.random.default_rng(500 + seed)
        Dc = corrupt_deleted(D1, assign, rng)
        variants = {"none": Dc, "deoffset": None, "mds": mds_impute(Dc, assign)}
        # deoffset: quantile gap is meaningless for noise, apply anyway for symmetry
        same = assign[:, None] == assign[None, :]
        inter = Dc[~same & ~np.eye(N, dtype=bool)]
        intra = Dc[same & ~np.eye(N, dtype=bool)]
        gap_hat = max(np.median(inter) - np.median(intra), 0.0)
        variants["deoffset"] = normalize_diameter(
            np.where(same, Dc, np.clip(Dc - gap_hat, 0, None)))
        for name, Dv in variants.items():
            T, _ = solve_egw_gpu(Dv, D2, eps=EPS, max_iter=250, tol=1e-8)
            p1 = neighborhood_precision(T, Z2, gt, unit(N), 1)
            p10 = neighborhood_precision(T, Z2, gt, unit(N), 10)
            rows.append(dict(seed=seed, corr=name, prec1=p1, prec10=p10,
                             rel10=float(p10 / b10)))
            print(f"[J] seed={seed} {name:<9}: prec1={p1:.3f} prec10={p10:.3f} "
                  f"rel={p10/b10:.2f} ({time.time()-t0:.0f}s)", flush=True)
    with open("results/exp_j.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print("saved results/exp_j.json")


if __name__ == "__main__":
    main()
