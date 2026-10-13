"""Exp I2: Within-instance noise sweep — isolates the effect of sigma on GW
correspondence while holding the warp instance fixed.

Exp I showed that across-instance variation dominates a naive SNR collapse.
Here each warp instance (fixed generator seed) is evaluated across an 8-step
sigma ladder, revealing the within-instance degradation curve.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, make_aligned_pair,
                           neighborhood_precision, normalize_diameter,
                           solve_egw_gpu, unit)
from exp_i_identifiability import knn_spacing

SIGMAS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.4]
SEEDS = [0, 1, 2]
N = 400
EPS = 0.03


def main():
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        for sig in SIGMAS:
            # base warp fixed by seed=100+seed; noise applied manually
            Z1, Z2b, gt = make_aligned_pair(n=N, d1=12, d2=7, seed=100 + seed,
                                            distortion=0.3, noise=0.0)
            if sig > 0:
                rng = np.random.default_rng(9000 + seed)
                Z2 = Z2b + sig * np.abs(Z2b).mean() * rng.normal(size=Z2b.shape)
            else:
                Z2 = Z2b
            D1 = normalize_diameter(euclidean_distance_matrix(Z1))
            D2 = normalize_diameter(euclidean_distance_matrix(Z2))
            T, log = solve_egw_gpu(D1, D2, eps=EPS, max_iter=250, tol=1e-8,
                                   device="cpu")
            p1 = neighborhood_precision(T, Z2, gt, unit(N), 1)
            p10 = neighborhood_precision(T, Z2, gt, unit(N), 10)
            h = knn_spacing(Z2)
            snr = sig * np.abs(Z2b).mean() / h
            rows.append(dict(seed=seed, sigma=sig, snr=float(snr), prec1=p1,
                             prec10=p10, egw=log["loss"]))
            print(f"[I2] seed={seed} sig={sig}: SNR={snr:.2f} prec1={p1:.3f} "
                  f"prec10={p10:.3f} ({time.time()-t0:.0f}s)", flush=True)
            with open("results/exp_i2.json", "w") as f:
                json.dump(rows, f, indent=2, default=float)
    print("saved results/exp_i2.json")
    for r in rows:
        print(f"seed={r['seed']} sigma={r['sigma']:<5} prec10={r['prec10']:.3f}")


if __name__ == "__main__":
    main()
