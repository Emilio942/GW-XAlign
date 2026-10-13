"""Exp G2: ordinal-preserving de-offset correction of fragmented distance
matrices (persisted + plotted). Corrections compared:
  none         : fragmented D as-is
  clip/impute  : value-level corrections (Exp G; destroy ordinal structure)
  deoff-est    : subtract the estimated inter/intra quantile offset from all
                 inter-expert entries (ordinal preserving, unsupervised)
"""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, make_aligned_pair,
                           neighborhood_precision, normalize_diameter,
                           solve_egw_gpu, unit)
from exp_f_moe import fragment


def main():
    t0 = time.time()
    N, K, EPS = 400, 8, 0.03
    GAPS = [0.3, 0.6]
    SEEDS = [0, 1, 2]
    rows = []
    for seed in SEEDS:
        Z1, Z2, gt = make_aligned_pair(n=N, d1=12, d2=7, seed=100 + seed,
                                       distortion=0.3, noise=0.05)
        D2 = normalize_diameter(euclidean_distance_matrix(Z2))
        ang = np.arctan2(Z1[:, 2], Z1[:, 0])
        assign = np.digitize(ang, np.linspace(-np.pi, np.pi, K + 1))
        D1 = normalize_diameter(euclidean_distance_matrix(Z1))
        # clean baseline
        T, _ = solve_egw_gpu(D1, D2, eps=EPS, max_iter=250, tol=1e-8)
        base10 = neighborhood_precision(T, Z2, gt, unit(N), 10)
        base1 = neighborhood_precision(T, Z2, gt, unit(N), 1)
        print(f"[G2] seed={seed} CLEAN: prec1={base1:.3f} prec10={base10:.3f}",
              flush=True)
        rows.append(dict(seed=seed, gap=0.0, corr="clean", prec1=base1,
                         prec10=base10, rel10=1.0))
        for gap in GAPS:
            Df = normalize_diameter(fragment(D1, assign, gap))
            same = assign[:, None] == assign[None, :]
            inter = Df[~same & ~np.eye(N, dtype=bool)]
            intra = Df[same & ~np.eye(N, dtype=bool)]
            gap_hat = max(np.quantile(inter, 0.5) - np.quantile(intra, 0.5), 0.0)
            Dc = np.where(same, Df, np.clip(Df - gap_hat, 0, None))
            Dc = normalize_diameter(Dc)
            for corr, Dv in (("none", Df), ("deoff-est", Dc)):
                T, _ = solve_egw_gpu(Dv, D2, eps=EPS, max_iter=250, tol=1e-8)
                p1 = neighborhood_precision(T, Z2, gt, unit(N), 1)
                p10 = neighborhood_precision(T, Z2, gt, unit(N), 10)
                rows.append(dict(seed=seed, gap=gap, corr=corr, prec1=p1,
                                 prec10=p10, rel10=float(p10 / base10)))
                print(f"[G2] seed={seed} gap={gap} {corr:<9}: prec1={p1:.3f} "
                      f"prec10={p10:.3f} rel={p10/base10:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    with open("results/exp_g2.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)

    # figure: prec@10 relative to each seed's clean baseline
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    width = 0.12
    xs = np.arange(len(SEEDS))
    for i, (gap, corr) in enumerate([(0.3, "none"), (0.3, "deoff-est"),
                                     (0.6, "none"), (0.6, "deoff-est")]):
        vals = [r["rel10"] for r in rows
                if r["gap"] == gap and r["corr"] == corr]
        ax.bar(xs + (i - 1.5) * width, vals, width,
               label=f"gap={gap}, {corr.replace('deoff-est','de-offset')}")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"instance {s}" for s in SEEDS])
    ax.set_ylabel("prec@10 relative to clean baseline")
    ax.set_title("Fragmentation repair (Exp G2): de-offset vs no correction\n"
                 "(unsupervised; dashed line = clean performance)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/exp_g_fragfix.png")
    print("saved results/exp_g2.json + figures/exp_g_fragfix.png")


if __name__ == "__main__":
    main()
