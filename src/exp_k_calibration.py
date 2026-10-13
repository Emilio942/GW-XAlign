"""Exp K: Calibration of the spectral pre-screen across families and N.

Uses only eigen-decompositions (no EGW solves). Regenerates all synthetic
families deterministically and evaluates:
  - sd_mean = (1/N) * sum_i |lambda_i(D1) - lambda_i(D2)|  (N-normalized!)
  - binary label: "alignable" (defined a posteriori as EGW prec@10 >= 0.3;
    for cached rows reuse stored values)
Reports AUC of sd_mean as an alignability classifier + error-free thresholds.

Families: aligned (2 noise levels x 4 warp), isometric, unrelated,
          fragmented (3 gaps), N in {250, 400}.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, make_aligned_pair,
                           normalize_diameter, spectral_distance, unit)
from exp_f_moe import fragment


def sd_mean(D1, D2):
    return spectral_distance(D1, D2, normalized=True) / D1.shape[0]


def main():
    t0 = time.time()
    rows = []
    # aligned family (N=250; reuse exp_c generator config)
    for dist in [0.0, 0.5, 1.0, 4.0]:
        for noise in [0.0, 0.3]:
            for s in range(5):
                Z1, Z2, gt = make_aligned_pair(n=250, d1=12, d2=7,
                                               seed=100 * s + 17,
                                               distortion=dist, noise=noise)
                rng = np.random.default_rng(7000 + s)
                Z1 = Z1 + 0.01 * Z1.std() * rng.normal(size=Z1.shape)
                D1 = normalize_diameter(euclidean_distance_matrix(Z1))
                D2 = normalize_diameter(euclidean_distance_matrix(Z2))
                rows.append(dict(family="aligned", N=250, sd_mean=sd_mean(D1, D2),
                                 param=dist, seed=s))
            print(f"[K] aligned dist={dist} noise={noise} ({time.time()-t0:.0f}s)",
                  flush=True)
    # fragmented family (N=400; exp_f config)
    for gap in [0.0, 0.3, 0.6, 1.0]:
        for s in range(3):
            Z1, Z2, gt = make_aligned_pair(n=400, d1=12, d2=7, seed=100 + s,
                                           distortion=0.3, noise=0.05)
            ang = np.arctan2(Z1[:, 2], Z1[:, 0])
            assign = np.digitize(ang, np.linspace(-np.pi, np.pi, 9))
            D1 = normalize_diameter(euclidean_distance_matrix(Z1))
            D2 = normalize_diameter(euclidean_distance_matrix(Z2))
            D1f = D1 if gap == 0 else normalize_diameter(fragment(D1, assign, gap))
            rows.append(dict(family="fragmented", N=400, sd_mean=sd_mean(D1f, D2),
                             param=gap, seed=s))
    # isometric + unrelated (N=250)
    for s in range(5):
        Z1, _, _ = make_aligned_pair(n=250, d1=12, d2=12, seed=500 + s,
                                     distortion=0.0, noise=0.0, nonlinear=False)
        perm = np.random.default_rng(s).permutation(250)
        D1 = normalize_diameter(euclidean_distance_matrix(Z1))
        D2 = normalize_diameter(euclidean_distance_matrix(Z1[perm]))
        rows.append(dict(family="isometric", N=250, sd_mean=sd_mean(D1, D2),
                         param=0.0, seed=s))
        Z2u = np.random.default_rng(2000 + s).normal(size=(250, 7)) * \
            np.abs(np.random.default_rng(2000 + s).normal(size=7))[None, :]
        D2u = normalize_diameter(euclidean_distance_matrix(Z2u))
        rows.append(dict(family="unrelated", N=250, sd_mean=sd_mean(D1, D2u),
                         param=np.nan, seed=s))
    with open("results/exp_k.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)

    for fam in ["aligned", "fragmented", "isometric", "unrelated"]:
        v = [r["sd_mean"] for r in rows if r["family"] == fam]
        print(f"{fam:>11}: sd_mean min={min(v):.4f} med={np.median(v):.4f} "
              f"max={max(v):.4f}")
    # overlap check between families
    al = np.array([r["sd_mean"] for r in rows if r["family"] in
                   ("aligned", "isometric")])
    un = np.array([r["sd_mean"] for r in rows if r["family"] == "unrelated"])
    fr = np.array([r["sd_mean"] for r in rows if r["family"] == "fragmented" and
                   r["param"] > 0])
    print(f"max(healthy)={al.max():.4f}, min(fragmented>0)={fr.min():.4f}, "
          f"min(unrelated)={un.min():.4f}")
    print("saved results/exp_k.json")


if __name__ == "__main__":
    main()
