"""Exp I3: Density effect at FIXED warp instance.

Earlier the N=200 -> 800 precision collapse was confounded with a change of
the random warp instance (generator seed). Here a single 800-point instance
is generated once and subset to N in {200,400,800} (first N rows, gt
restricted accordingly), holding warp, noise and geometry fixed. If
precision drops with N, fingerprint collision (density) is real; if not,
the earlier collapse was an instance artifact.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, make_aligned_pair,
                           neighborhood_precision, normalize_diameter,
                           solve_egw_gpu, unit)

EPS = 0.03
SEEDS = [0, 1, 2]
N_FULL = 800
SUBS = [200, 400, 800]


def main():
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        Z1f, Z2f, gtf = make_aligned_pair(n=N_FULL, d1=12, d2=7,
                                          seed=100 + seed, distortion=0.3,
                                          noise=0.05)
        for N in SUBS:
            Z1, Z2, gt = Z1f[:N], Z2f[:N], gtf[:N]
            D1 = normalize_diameter(euclidean_distance_matrix(Z1))
            D2 = normalize_diameter(euclidean_distance_matrix(Z2))
            T, log = solve_egw_gpu(D1, D2, eps=EPS, max_iter=200, tol=1e-8,
                                   device="cpu")
            p1 = neighborhood_precision(T, Z2, gt, unit(N), 1)
            p10 = neighborhood_precision(T, Z2, gt, unit(N), 10)
            rows.append(dict(seed=seed, N=N, prec1=p1, prec10=p10,
                             egw=log["loss"]))
            print(f"[I3] seed={seed} N={N}: prec1={p1:.3f} prec10={p10:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            with open("results/exp_i3.json", "w") as f:
                json.dump(rows, f, indent=2, default=float)
    print("saved results/exp_i3.json")
    for N in SUBS:
        v = [r["prec10"] for r in rows if r["N"] == N]
        print(f"N={N}: mean prec10 = {np.mean(v):.3f}  (per-seed {np.round(v,3)})")


if __name__ == "__main__":
    main()
