"""Exp H2: Debiased EGW on CIFAR feature clouds (GPU, N=500)."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, normalize_diameter,
                           solve_egw_gpu)

N_ALIGN = 500
EPS_LIST = [0.005, 0.02]
PAIRS = [("ViT_S", "Mixer_S"), ("ViT_S", "CNN_S"), ("Mixer_S", "CNN_S"),
         ("ViT_S", "ViT_S_s1"), ("CNN_S", "CNN_S_s1")]

L = json.load(open("results/exp_l.json"))
res = L["results"] if isinstance(L, dict) else L
E = {r["tag"] + "|" + str(r["eps"]): r["acc_align"] for r in res}


def main():
    t0 = time.time()
    rows = []
    cache = {}
    for a, b in PAIRS:
        da = np.load(f"data/cifar/{a}.npz"); db = np.load(f"data/cifar/{b}.npz")
        ia = np.random.default_rng(1).choice(len(da["Z"]), N_ALIGN, replace=False)
        ib = np.random.default_rng(2).choice(len(db["Z"]), N_ALIGN, replace=False)
        ZA, ZB = da["Z"][ia], db["Z"][ib]
        D1 = normalize_diameter(euclidean_distance_matrix(ZA))
        D2 = normalize_diameter(euclidean_distance_matrix(ZB))
        for eps in EPS_LIST:
            if (a, eps) not in cache:
                _, la = solve_egw_gpu(D1, D1, eps=eps, max_iter=150, tol=1e-8)
                cache[(a, eps)] = la["loss"]
            if (b, eps) not in cache:
                _, lb = solve_egw_gpu(D2, D2, eps=eps, max_iter=150, tol=1e-8)
                cache[(b, eps)] = lb["loss"]
            _, lp = solve_egw_gpu(D1, D2, eps=eps, max_iter=200, tol=1e-8)
            deb = lp["loss"] - 0.5 * cache[(a, eps)] - 0.5 * cache[(b, eps)]
            acc = E[f"{a}->{b}|{eps}"]
            rows.append(dict(pair=f"{a}->{b}", eps=eps, egw=lp["loss"],
                             debiased=float(deb), acc_align=acc))
            print(f"[H2] {a}->{b} eps={eps}: egw={lp['loss']:.4f} deb={deb:.4f} "
                  f"acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    from scipy.stats import spearmanr
    r_raw = spearmanr([r["egw"] for r in rows], [r["acc_align"] for r in rows])
    r_deb = spearmanr([r["debiased"] for r in rows], [r["acc_align"] for r in rows])
    with open("results/exp_h2.json", "w") as f:
        json.dump(dict(rows=rows,
                       correlations=dict(raw=float(r_raw.statistic),
                                         debiased=float(r_deb.statistic))),
                  f, indent=2, default=float)
    print(f"rho(raw EGW, acc)      = {r_raw.statistic:+.3f}")
    print(f"rho(debiased EGW, acc) = {r_deb.statistic:+.3f}")


if __name__ == "__main__":
    main()
