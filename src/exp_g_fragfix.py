"""Exp G: Correcting routing fragmentation BEFORE solving EGW (follow-up to F).

If Exp F showed that entropic smoothing cannot repair a fragmented distance
matrix, the fix should operate on D1 itself. Two cheap corrections:

  clip:    D_corr[i,j] = min(D_frag[i,j], q90_intra)  for inter-expert pairs
  impute:  inter-expert entries replaced by the median intra-expert distance

Success criterion: recover a substantial fraction of the clean (gap=0)
prec@10 baseline across the fragmentation grid.
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


def correct(D_frag, assign, mode):
    D = D_frag.copy()
    same = assign[:, None] == assign[None, :]
    intra = D[same & ~np.eye(len(assign), dtype=bool)]
    if mode == "clip":
        q = np.quantile(intra, 0.90)
        D = np.where(same, D, np.minimum(D, q))
    elif mode == "impute":
        m = np.median(intra)
        D = np.where(same, D, m)
    return D


def main():
    t0 = time.time()
    N = 400
    GAPS = [0.0, 0.3, 0.6, 1.0]
    EPS_LIST = [0.01, 0.03, 0.1]
    K = 8
    SEEDS = [0, 1, 2]
    rows = []
    for seed in SEEDS:
        Z1, Z2, gt = make_aligned_pair(n=N, d1=12, d2=7, seed=100 + seed,
                                       distortion=0.3, noise=0.05)
        D2 = normalize_diameter(euclidean_distance_matrix(Z2))
        ang = np.arctan2(Z1[:, 2], Z1[:, 0])
        assign = np.digitize(ang, np.linspace(-np.pi, np.pi, K + 1))
        D1 = normalize_diameter(euclidean_distance_matrix(Z1))
        for gap in GAPS:
            variants = {"none": D1 if gap == 0 else
                        normalize_diameter(fragment(D1, assign, gap))}
            if gap > 0:
                Df = normalize_diameter(fragment(D1, assign, gap))
                for mode in ("clip", "impute"):
                    variants[mode] = normalize_diameter(correct(Df, assign, mode))
            for vname, Dv in variants.items():
                for eps in EPS_LIST:
                    T, log = solve_egw_gpu(Dv, D2, eps=eps, max_iter=250, tol=1e-8)
                    p1 = neighborhood_precision(T, Z2, gt, k=1)
                    p10 = neighborhood_precision(T, Z2, gt, k=10)
                    rows.append(dict(seed=seed, gap=gap, corr=vname, eps=eps,
                                     egw=log["loss"], prec1=p1, prec10=p10))
                    print(f"[G] seed={seed} gap={gap} corr={vname:<6} eps={eps}: "
                          f"prec1={p1:.3f} prec10={p10:.3f} ({time.time()-t0:.0f}s)",
                          flush=True)

    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["gap"], r["corr"], r["eps"])].append((r["prec1"], r["prec10"]))
    table = {f"gap={g},corr={c},eps={e}": dict(
        prec1=float(np.mean([x[0] for x in v])),
        prec10=float(np.mean([x[1] for x in v])))
        for (g, c, e), v in agg.items()}

    with open("results/exp_g.json", "w") as f:
        json.dump(dict(rows=rows, aggregate=table,
                       config=dict(N=N, GAPS=GAPS, EPS_LIST=EPS_LIST, K=K)),
                  f, indent=2, default=float)
    print("\n=== aggregated (mean prec@1 / prec@10) ===")
    for k, v in sorted(table.items()):
        print(f"{k:>30}: prec1={v['prec1']:.3f} prec10={v['prec10']:.3f}")
    print("saved results/exp_g.json")


if __name__ == "__main__":
    main()
