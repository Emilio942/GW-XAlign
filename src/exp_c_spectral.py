"""Exp C: Spectral predictor of alignability (Research Question 2).

Q:  Does  sum_i |lambda_i(D1) - lambda_i(D2)|  (eigenspectrum distance of the
diameter-normalized distance matrices) predict, gradient-free and BEFORE
solving the O(N^2 M + N M^2) EGW problem,
   (a) the EGW / GW divergence, and
   (b) the zero-shot correspondence quality (neighborhood precision)?

Controls: isometric-but-permuted pairs (spectral distance = 0, precision = 1);
unrelated structurally mismatched pairs.
"""
import json
import sys
import time
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, make_aligned_pair,
                           neighborhood_precision, normalize_diameter,
                           solve_egw_gpu, solve_gw, spectral_distance,
                           spectral_distance_wasserstein, unit)

N = 250
EPS = 0.02
MAX_ITER = 200


def metrics_for_pair(Z1, Z2, gt):
    D1 = normalize_diameter(euclidean_distance_matrix(Z1))
    D2 = normalize_diameter(euclidean_distance_matrix(Z2))
    sd_l1 = spectral_distance(D1, D2, normalized=True)
    sd_w1 = spectral_distance_wasserstein(D1, D2, normalized=True)
    T, log = solve_egw_gpu(D1, D2, eps=EPS, max_iter=MAX_ITER)
    p1 = neighborhood_precision(T, Z2, gt, k=1)
    p10 = neighborhood_precision(T, Z2, gt, k=10)
    _, log_gw = solve_gw(D1, D2, num_iter=200)
    return dict(spectral_l1=sd_l1, spectral_w1=sd_w1, egw=log["loss"],
                gw=log_gw["loss"], prec1=p1, prec10=p10)


def main():
    t0 = time.time()
    DISTORTIONS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    NOISES = [0.0, 0.1, 0.3]
    SEEDS = [0, 1, 2, 3, 4]
    rows = []

    for dist in DISTORTIONS:
        for noise in NOISES:
            for s in SEEDS:
                Z1, Z2, gt = make_aligned_pair(n=N, d1=12, d2=7,
                                               seed=100 * s + 17,
                                               distortion=dist, noise=noise)
                # small jitter on Z1: breaks exact manifold degeneracies
                rng = np.random.default_rng(7000 + s)
                Z1 = Z1 + 0.01 * Z1.std() * rng.normal(size=Z1.shape)
                m = metrics_for_pair(Z1, Z2, gt)
                m.update(family="aligned", distortion=dist, noise=noise, seed=s)
                rows.append(m)
            print(f"[C] distortion={dist} noise={noise} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # isometric control
    for s in SEEDS:
        Z1, _, _ = make_aligned_pair(n=N, d1=12, d2=12, seed=500 + s,
                                     distortion=0.0, noise=0.0, nonlinear=False)
        perm = np.random.default_rng(s).permutation(N)
        Z2p, gt = Z1[perm], np.empty(N, int)
        gt[perm] = np.arange(N)
        m = metrics_for_pair(Z1, Z2p, gt)
        m.update(family="isometric", distortion=0.0, noise=0.0, seed=s)
        rows.append(m)

    # negative control: unrelated clouds
    for s in SEEDS:
        Z1, _, _ = make_aligned_pair(n=N, d1=12, d2=12, seed=900 + s,
                                     distortion=0.0, noise=0.0)
        rng = np.random.default_rng(2000 + s)
        Z2u = rng.normal(size=(N, 7)) * np.abs(rng.normal(size=7))[None, :]
        gt = np.arange(N)  # arbitrary
        m = metrics_for_pair(Z1, Z2u, gt)
        m.update(family="unrelated", distortion=np.nan, noise=np.nan, seed=s)
        rows.append(m)

    al = [r for r in rows if r["family"] == "aligned"]

    def rho(xs, ys):
        r = spearmanr(xs, ys)
        return dict(rho=float(r.statistic), p=float(r.pvalue))

    out = dict(rows=rows, config=dict(N=N, eps=EPS), correlations={
        "spectral_l1 vs (1-prec1)": rho([r["spectral_l1"] for r in al],
                                        [1 - r["prec1"] for r in al]),
        "spectral_l1 vs (1-prec10)": rho([r["spectral_l1"] for r in al],
                                         [1 - r["prec10"] for r in al]),
        "spectral_w1 vs (1-prec10)": rho([r["spectral_w1"] for r in al],
                                         [1 - r["prec10"] for r in al]),
        "spectral_l1 vs gw": rho([r["spectral_l1"] for r in al],
                                 [r["gw"] for r in al]),
        "spectral_w1 vs egw": rho([r["spectral_w1"] for r in al],
                                  [r["egw"] for r in al]),
        "gw vs (1-prec10)": rho([r["gw"] for r in al],
                                [1 - r["prec10"] for r in al]),
        "egw vs (1-prec10)": rho([r["egw"] for r in al],
                                 [1 - r["prec10"] for r in al]),
    })
    with open("results/exp_c.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    for k, v in out["correlations"].items():
        print(f"rho[{k}] = {v['rho']:+.3f} (p={v['p']:.1e})")
    iso = [(round(r["spectral_l1"], 4), r["prec1"]) for r in rows
           if r["family"] == "isometric"]
    un = [(round(r["spectral_l1"], 3), round(r["gw"], 4)) for r in rows
          if r["family"] == "unrelated"]
    print("isometric controls (spec_l1, prec1):", iso)
    print("unrelated controls (spec_l1, gw):", un)
    print("saved results/exp_c.json")


if __name__ == "__main__":
    main()
