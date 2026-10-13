"""Exp F: Routing-induced fragmentation in MoE-like latent spaces (RQ1) and
the role of entropic smoothing eps.

Simulation:
  - Z2: smooth manifold observation (target architecture).
  - Z1: same latent manifold, but pairwise distances within/expert vs
    across/expert are distorted: D_frag[i,j] = D[i,j] if same expert else
    (1-gap)*D + gap  (gap=0: clean; gap=1: full disconnection = routing noise).

Questions tested:
  F1  How does fragmentation alter the spectrum of D1?
  F2  Does larger eps "collapse" the fragmentation noise (restore alignment),
      and what is the price paid on fine local structure (prec@1 vs prec@10)?
"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, make_aligned_pair,
                           neighborhood_precision, normalize_diameter,
                           solve_egw_gpu, spectral_distance,
                           spectral_distance_wasserstein, spectrum, unit)


def fragment(D, assign, gap):
    Df = D.copy()
    n = len(assign)
    same = assign[:, None] == assign[None, :]
    Df = np.where(same, Df, (1.0 - gap) * Df + gap)
    return Df


def main():
    t0 = time.time()
    N = 400
    GAPS = [0.0, 0.3, 0.6, 1.0]
    EPS_LIST = [0.01, 0.03, 0.1, 0.3]
    K_EXPERTS = 8
    SEEDS = [0, 1, 2]
    rows = []
    spec_rows = []

    for seed in SEEDS:
        Z1, Z2, gt = make_aligned_pair(n=N, d1=12, d2=7, seed=100 + seed,
                                       distortion=0.3, noise=0.05)
        D2 = normalize_diameter(euclidean_distance_matrix(Z2))
        # assign synthetic "experts" by angular sector of the roll (t param):
        # use the polar angle proxy = first coordinate sign structure of Z1
        ang = np.arctan2(Z1[:, 2], Z1[:, 0])
        assign = np.digitize(ang, np.linspace(-np.pi, np.pi, K_EXPERTS + 1))
        D1 = normalize_diameter(euclidean_distance_matrix(Z1))
        for gap in GAPS:
            D1f = D1 if gap == 0 else normalize_diameter(fragment(D1, assign, gap))
            spec_rows.append(dict(seed=seed, gap=gap,
                                  spectral_shift=spectral_distance(D1, D1f),
                                  wasserstein_shift=spectral_distance_wasserstein(D1, D1f),
                                  l2=float(np.linalg.norm(D1f - D1))))
            for eps in EPS_LIST:
                T, log = solve_egw_gpu(D1f, D2, eps=eps, max_iter=250, tol=1e-8)
                p1 = neighborhood_precision(T, Z2, gt, k=1)
                p10 = neighborhood_precision(T, Z2, gt, k=10)
                rows.append(dict(seed=seed, gap=gap, eps=eps, egw=log["loss"],
                                 prec1=p1, prec10=p10))
                print(f"[F] seed={seed} gap={gap} eps={eps}: "
                      f"prec1={p1:.3f} prec10={p10:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)

    # aggregate
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["gap"], r["eps"])].append((r["prec1"], r["prec10"], r["egw"]))
    table = {f"gap={g},eps={e}": dict(
        prec1=float(np.mean([x[0] for x in v])),
        prec10=float(np.mean([x[1] for x in v])),
        egw=float(np.mean([x[2] for x in v]))) for (g, e), v in agg.items()}

    out = dict(rows=rows, spec_rows=spec_rows, aggregate=table,
               config=dict(N=N, GAPS=GAPS, EPS_LIST=EPS_LIST, K=K_EXPERTS))
    with open("results/exp_f.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n=== aggregated (mean over seeds) ===")
    for k, v in sorted(table.items()):
        print(f"{k:>18}: prec1={v['prec1']:.3f} prec10={v['prec10']:.3f} "
              f"egw={v['egw']:.4f}")
    print("spectral shift by gap:",
          {g: round(float(np.mean([s['spectral_shift'] for s in spec_rows if s['gap']==g])),3) for g in GAPS})
    print("saved results/exp_f.json")


if __name__ == "__main__":
    main()
