"""Exp A: Empirical concentration of the entropic GW divergence.

Tests the hypothesis (Cycle 1):  |EGW_eps(hat_mu_N, hat_nu_N) - EGW_eps(mu, nu)|
= O( sqrt(log N / (N eps)) + eps ), i.e. log-log slope ~ -1/2 in N, and
scaling ~ eps^{-1/2} at fixed N (plus an O(eps) bias vs unregularized GW).

Population proxy: one large reference cloud (N_ref) from the same generator.
"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (cosine_distance_matrix, euclidean_distance_matrix,
                           make_aligned_pair, normalize_diameter, solve_egw,
                           solve_egw_gpu, solve_gw)

RNG_SEED = 1234
N_REF = 2000
N_GRID = [100, 200, 400, 800]
EPS_GRID = [0.01, 0.03, 0.1]
R = 8          # repetitions per (N, eps); reduced for large N below
R_ADAPTIVE = {100: 8, 200: 8, 400: 5, 800: 4}
D1, D2 = 12, 7
MAX_ITER = 200
TOL = 1e-8


def pair_C(N, seed):
    Z1, Z2, _ = make_aligned_pair(n=N, d1=D1, d2=D2, seed=seed,
                                  distortion=0.3, noise=0.05)
    return (normalize_diameter(euclidean_distance_matrix(Z1)),
            normalize_diameter(euclidean_distance_matrix(Z2)))


def egw_value(N, eps, seed):
    C1, C2 = pair_C(N, seed)
    _, log = solve_egw_gpu(C1, C2, eps=eps, max_iter=MAX_ITER, tol=TOL)
    return log["loss"]


def main():
    t0 = time.time()
    # Population proxy
    egw_pop = {eps: egw_value(N_REF, eps, 99_000) for eps in EPS_GRID}
    print("pop done", egw_pop, f"({time.time()-t0:.0f}s)", flush=True)
    # Unregularized GW on the reference (bias anchor as eps -> 0)
    C1, C2 = pair_C(N_REF, 99_000)
    _, log_gw = solve_gw(C1, C2, num_iter=300)
    gw_pop = log_gw["loss"]
    print(f"GW_pop={gw_pop:.5f}", flush=True)

    rows = []
    for eps in EPS_GRID:
        for N in N_GRID:
            devs = []
            for r in range(R_ADAPTIVE[N]):
                v = egw_value(N, eps, RNG_SEED + 1000 * N + r)
                devs.append(abs(v - egw_pop[eps]))
            devs = np.array(devs)
            rows.append(dict(eps=eps, N=N, mean_dev=float(devs.mean()),
                             median_dev=float(np.median(devs)),
                             std_dev=float(devs.std()), reps=len(devs)))
            print(f"[{time.time()-t0:7.1f}s] eps={eps:<6} N={N:<5} "
                  f"dev={devs.mean():.5f}", flush=True)

    fits = {}
    for eps in EPS_GRID:
        xs = np.log(np.array([r["N"] for r in rows if r["eps"] == eps]))
        ys = np.log(np.array([r["mean_dev"] for r in rows if r["eps"] == eps]))
        A = np.vstack([xs, np.ones_like(xs)]).T
        alpha, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        fits[str(eps)] = dict(alpha=float(alpha), intercept=float(c))

    eps_scaling = {}
    for N in N_GRID:
        vals = np.array([r["mean_dev"] for r in rows if r["N"] == N])
        xs = np.log(np.array(EPS_GRID))
        ys = np.log(vals)
        A = np.vstack([xs, np.ones_like(xs)]).T
        beta, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        eps_scaling[str(N)] = dict(beta=float(beta))

    out = dict(config=dict(N_REF=N_REF, N_GRID=N_GRID, EPS_GRID=EPS_GRID,
                           R=R, d1=D1, d2=D2, max_iter=MAX_ITER),
               egw_pop=egw_pop, gw_pop=gw_pop, rows=rows,
               powerlaw_slope_alpha=fits,
               eps_scaling_beta=eps_scaling)
    with open("results/exp_a.json", "w") as f:
        json.dump(out, f, indent=2)
    print("slope alpha in N (theory: -0.5):", fits)
    print("slope beta in eps:", eps_scaling)
    print("saved results/exp_a.json")


if __name__ == "__main__":
    main()
