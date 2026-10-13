"""Exp B: Lipschitz stability of the optimal EGW coupling under metric
perturbations.   Exp D: stability of the barycentric pushforward under
coupling perturbations.

Cycle 1-hyp2 / Cycle 8 / Cycle 9 claims tested:
  B:  ||gamma*(D1) - gamma*(D1+Delta)||_F  =  L(eps) * ||Delta||_F + o(Delta),
      with L(eps) ~ C / eps.
  D:  ||T_gamma - T_{gamma'}||  =  K * ||gamma - gamma'||_F  (linear regime),
      with K depending on eps.
"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, unit, solve_egw_gpu,
                           make_aligned_pair, normalize_diameter, solve_egw,
                           unit)


def perturb_points(Z, delta, rng):
    Zp = Z + delta * rng.normal(size=Z.shape) / np.sqrt(Z.shape[1])
    return Zp


def exp_b():
    rng = np.random.default_rng(7)
    N = 300
    Z1, Z2, _ = make_aligned_pair(n=N, d1=12, d2=7, seed=42,
                                  distortion=0.3, noise=0.05)
    C1 = normalize_diameter(euclidean_distance_matrix(Z1))
    C2 = normalize_diameter(euclidean_distance_matrix(Z2))
    EPS = [0.005, 0.02, 0.05]
    DELTAS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    R = 3
    rows = []
    for eps in EPS:
        T0, log0 = solve_egw_gpu(C1, C2, eps=eps, max_iter=300, tol=1e-8)
        for d in DELTAS:
            diffs = []
            for r in range(R):
                Z1p = perturb_points(Z1, d, rng)
                C1p = normalize_diameter(euclidean_distance_matrix(Z1p))
                Tp, _ = solve_egw_gpu(C1p, C2, eps=eps, max_iter=300, tol=1e-8)
                diffs.append(float(np.linalg.norm(Tp - T0)))
            rows.append(dict(eps=eps, delta=d, mean_diff=float(np.mean(diffs)),
                             std=float(np.std(diffs))))
            print(f"[B] eps={eps:<6} delta={d:<6} ||dT||_F={np.mean(diffs):.5f}",
                  flush=True)
    # slopes
    slopes = {}
    for eps in EPS:
        xs = np.array([r["delta"] for r in rows if r["eps"] == eps])
        ys = np.array([r["mean_diff"] for r in rows if r["eps"] == eps])
        # slope through origin (Lipschitz constant estimate)
        slope = float(np.sum(xs * ys) / np.sum(xs * xs))
        slopes[str(eps)] = slope
    # check L(eps) ~ 1/eps : fit log L vs log eps
    le = np.log(np.array(EPS)); ls = np.log(np.array([slopes[str(e)] for e in EPS]))
    A = np.vstack([le, np.ones_like(le)]).T
    beta, c = np.linalg.lstsq(A, ls, rcond=None)[0]
    with open("results/exp_b.json", "w") as f:
        json.dump(dict(rows=rows, slopes=slopes, eps_exponent=float(beta)),
                  f, indent=2)
    print("B slopes:", slopes, "| dlogL/dlog(eps) =", float(beta))


def exp_d():
    rng = np.random.default_rng(11)
    N = 300
    Z1, Z2, _ = make_aligned_pair(n=N, d1=12, d2=7, seed=42,
                                  distortion=0.3, noise=0.05)
    C1 = normalize_diameter(euclidean_distance_matrix(Z1))
    C2 = normalize_diameter(euclidean_distance_matrix(Z2))
    Y = Z2
    p = unit(N)
    ETA = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    EPS = [0.005, 0.02, 0.05]
    rows = []
    for eps in EPS:
        T0, _ = solve_egw_gpu(C1, C2, eps=eps, max_iter=300, tol=1e-8)
        B0 = (T0 / p[:, None]) @ Y
        for eta in ETA:
            diffs, dTs = [], []
            for r in range(5):
                G = rng.normal(size=T0.shape)
                Tp = T0 + eta * G * T0.mean()   # relative-size perturbation
                Tp = np.clip(Tp, 1e-12, None)
                # project back to polytope approximately (row/col normalize twice)
                for _ in range(2):
                    Tp = Tp / Tp.sum(axis=1, keepdims=True) * p[:, None]
                    Tp = Tp / Tp.sum(axis=0, keepdims=True) * p[None, :]
                Bp = (Tp / p[:, None]) @ Y
                diffs.append(float(np.max(np.linalg.norm(Bp - B0, axis=1))))
                dTs.append(float(np.linalg.norm(Tp - T0)))
            rows.append(dict(eps=eps, eta=eta, mean_bary_sup=float(np.mean(diffs)),
                             mean_dT=float(np.mean(dTs))))
            print(f"[D] eps={eps:<6} eta={eta:<5} sup||dB||={np.mean(diffs):.5f}",
                  flush=True)
    slopes = {}
    for eps in EPS:
        xs = np.array([r["eta"] for r in rows if r["eps"] == eps])
        ys = np.array([r["mean_bary_sup"] for r in rows if r["eps"] == eps])
        slopes[str(eps)] = float(np.sum(xs * ys) / np.sum(xs * xs))
    with open("results/exp_d.json", "w") as f:
        json.dump(dict(rows=rows, slopes=slopes), f, indent=2)
    print("D slopes (barycentric sup change per coupling perturbation):", slopes)


if __name__ == "__main__":
    t = time.time()
    exp_b()
    print(f"exp B done in {time.time()-t:.1f}s")
    exp_d()
    print(f"exp D done in {time.time()-t:.1f}s")
