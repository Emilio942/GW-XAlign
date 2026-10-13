"""Exp I: The identifiability law of GW correspondence.

Hypothesis (from §9 of the report): individual correspondence under GW is
identified only when the observation noise sigma is small relative to the
local k-NN spacing h_N of the manifold sample. Prediction: correspondence
precision collapses as a function of SNR = sigma / h_N, with all (N, sigma)
configurations falling on ONE universal collapse curve  prec = f(sigma/h_N).

Design: swiss-roll pairs, fixed warp 0.3; N in {100,200,400,800};
noise sigma in {0.02,0.05,0.15} (fraction of mean|Z2|); eps=0.03.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, make_aligned_pair,
                           neighborhood_precision, normalize_diameter,
                           solve_egw_gpu, unit)

N_GRID = [100, 200, 400, 800]
SIGMAS = [0.02, 0.05, 0.15]
SEEDS = [0, 1, 2]
EPS = 0.03
K_NN = 10


def knn_spacing(Z, k=K_NN):
    from scipy.spatial.distance import cdist
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    return float(np.median(np.sort(D, axis=1)[:, k - 1]))


def main():
    t0 = time.time()
    rows = []
    for N in N_GRID:
        for sig in SIGMAS:
            for seed in SEEDS:
                Z1, Z2, gt = make_aligned_pair(n=N, d1=12, d2=7,
                                               seed=3000 + 100 * seed + N,
                                               distortion=0.3, noise=sig)
                rng = np.random.default_rng(7000 + seed)
                Z1 = Z1 + 0.01 * Z1.std() * rng.normal(size=Z1.shape)
                D1 = normalize_diameter(euclidean_distance_matrix(Z1))
                D2 = normalize_diameter(euclidean_distance_matrix(Z2))
                h = knn_spacing(Z2 / (np.linalg.norm(Z2) / np.sqrt(N)))  # spacing
                # normalize noise to the same diameter units as prec space:
                h_norm = knn_spacing(Z2) / np.abs(Z2).mean()
                s_norm = (sig * np.abs(Z2).mean() * np.sqrt(Z2.shape[1])) / (
                    np.sqrt(Z2.shape[1]) * np.abs(Z2).mean())
                snr = (sig * np.sqrt(Z2.shape[1])) / max(h_norm * np.sqrt(Z2.shape[1]), 1e-12)
                # simpler, well-defined SNR: sigma_abs / h_abs
                sigma_abs = sig * np.abs(Z2).mean()
                h_abs = knn_spacing(Z2)
                snr = sigma_abs / h_abs
                it = 150 if N >= 800 else 250
                T, log = solve_egw_gpu(D1, D2, eps=EPS, max_iter=it, tol=1e-8,
                                       device="cpu")
                p1 = neighborhood_precision(T, Z2, gt, unit(N), 1)
                p10 = neighborhood_precision(T, Z2, gt, unit(N), 10)
                rows.append(dict(N=N, sigma=sig, seed=seed, snr=snr,
                                 h_abs=h_abs, prec1=p1, prec10=p10,
                                 egw=log["loss"]))
                print(f"[I] N={N} sig={sig} seed={seed}: SNR={snr:.3f} "
                      f"prec1={p1:.3f} prec10={p10:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    with open("results/exp_i.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)

    # collapse check: bin prec10 by SNR and measure dispersion within bins
    snrs = np.array([r["snr"] for r in rows])
    precs = np.array([r["prec10"] for r in rows])
    order = np.argsort(snrs)
    sb, pb = snrs[order], precs[order]
    bins = np.array_split(np.arange(len(sb)), 8)
    print("SNR bin  mean_prec10  std  count")
    for b in bins:
        print(f"{sb[b].mean():7.3f}  {pb[b].mean():.3f}  {pb[b].std():.3f}  {len(b)}")
    print("saved results/exp_i.json")


if __name__ == "__main__":
    main()
