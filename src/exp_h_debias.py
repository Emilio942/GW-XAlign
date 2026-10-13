"""Exp H: Debiased EGW (Sinkhorn-divergence style) as alignment statistic.

  debias(eps) = EGW_eps(A,B) - 0.5*EGW_eps(A,A) - 0.5*EGW_eps(B,B)

Computed on the MNIST feature clouds (N=500, CPU) for the 5 architecture
pairs, and compared (Spearman) against kNN transfer accuracy from Exp E.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, normalize_diameter,
                           solve_egw_gpu, unit)

N_ALIGN = 500
EPS_LIST = [0.005, 0.02]
PAIRS = [("MLP128", "CNN64"), ("MLP128", "MLP16"), ("MLP128", "MLP128_s1"),
         ("CNN64", "CNN64_s1"), ("MLP16", "CNN64")]

# transfer accuracies from exp_e (eps -> per-pair acc_align)
E = {r["tag"] + "|" + str(r["eps"]): r["acc_align"] for r in json.load(open("results/exp_e.json"))["results"]}


def clouds(a, b):
    da = np.load(f"data/feats/{a}.npz"); db = np.load(f"data/feats/{b}.npz")
    ia = np.random.default_rng(1).choice(len(da["Z"]), N_ALIGN, replace=False)
    ib = np.random.default_rng(2).choice(len(db["Z"]), N_ALIGN, replace=False)
    return da["Z"][ia], db["Z"][ib]


def main():
    t0 = time.time()
    rows = []
    cache = {}
    for a, b in PAIRS:
        ZA, ZB = clouds(a, b)
        D1 = normalize_diameter(euclidean_distance_matrix(ZA))
        D2 = normalize_diameter(euclidean_distance_matrix(ZB))
        for eps in EPS_LIST:
            if (a, eps) not in cache:
                _, la = solve_egw_gpu(D1, D1, eps=eps, max_iter=150, tol=1e-8,
                                      device="cpu")
                cache[(a, eps)] = la["loss"]
            if (b, eps) not in cache:
                _, lb = solve_egw_gpu(D2, D2, eps=eps, max_iter=150, tol=1e-8,
                                      device="cpu")
                cache[(b, eps)] = lb["loss"]
            _, lp = solve_egw_gpu(D1, D2, eps=eps, max_iter=200, tol=1e-8,
                                  device="cpu")
            deb = lp["loss"] - 0.5 * cache[(a, eps)] - 0.5 * cache[(b, eps)]
            acc = E[f"{a}->{b}|{eps}"]
            rows.append(dict(pair=f"{a}->{b}", eps=eps, egw=lp["loss"],
                             egw_aa=cache[(a, eps)], egw_bb=cache[(b, eps)],
                             debiased=float(deb), acc_align=acc))
            print(f"[H] {a}->{b} eps={eps}: egw={lp['loss']:.4f} "
                  f"deb={deb:.4f} acc={acc:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        # write incrementally
        with open("results/exp_h.json", "w") as f:
            json.dump(rows, f, indent=2, default=float)

    from scipy.stats import spearmanr
    r_raw = spearmanr([r["egw"] for r in rows], [r["acc_align"] for r in rows])
    r_deb = spearmanr([r["debiased"] for r in rows], [r["acc_align"] for r in rows])
    out = dict(rows=rows,
               correlations=dict(raw=float(r_raw.statistic),
                                 debiased=float(r_deb.statistic)))
    with open("results/exp_h.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"rho(raw EGW, acc)     = {r_raw.statistic:+.3f}")
    print(f"rho(debiased EGW, acc) = {r_deb.statistic:+.3f}")
    print("saved results/exp_h.json")


if __name__ == "__main__":
    main()
