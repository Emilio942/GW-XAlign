"""Generate all figures for the GW-XAlign experiment suite."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})

# ---------------------------------------------------------------- Exp A
d = json.load(open("results/exp_a.json"))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for eps in d["config"]["EPS_GRID"]:
    rows = [r for r in d["rows"] if r["eps"] == eps]
    N = [r["N"] for r in rows]
    dev = [r["mean_dev"] for r in rows]
    err = [r["std_dev"] for r in rows]
    axes[0].errorbar(N, dev, yerr=err, marker="o", ms=3,
                     label=f"$\\varepsilon$={eps}")
    # theoretical guide: C N^{-1/2}
    c = dev[0] * np.sqrt(N[0])
    axes[0].plot(N, c / np.sqrt(N), "--", lw=0.8, color="gray")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("sample size $N$"); axes[0].set_ylabel(r"$|\,\widehat{EGW}_\varepsilon - EGW_\varepsilon(\mu_N^{ref})|$")
axes[0].set_title("Concentration of empirical EGW\n(dashed: $N^{-1/2}$ guide)")
axes[0].legend(fontsize=7)

Ns = d["config"]["N_GRID"]
alphas = [d["powerlaw_slope_alpha"][str(e)]["alpha"] for e in d["config"]["EPS_GRID"]]
gw = d["gw_pop"]
egws = [d["egw_pop"][str(e)] for e in d["config"]["EPS_GRID"]]
eps_list = d["config"]["EPS_GRID"]
axes[1].plot(eps_list, [abs(e - gw) for e in egws], "s-", label="bias $|EGW_\\varepsilon - GW|$ (ref)")
for N, col in zip(Ns, ["C0", "C1", "C2", "C3"]):
    vals = [r["mean_dev"] for r in d["rows"] if r["N"] == N]
    axes[1].plot(eps_list, vals, "o--", color=col, ms=3, label=f"dev @ N={N}")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel(r"regularization $\varepsilon$")
axes[1].set_ylabel("deviation / bias")
axes[1].set_title(f"$\\varepsilon$-scaling (slopes in N: {np.round(alphas,2)})")
axes[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig("figures/exp_a_concentration.png")
print("fig A saved")

# ---------------------------------------------------------------- Exp B/D
db = json.load(open("results/exp_b.json"))
dd = json.load(open("results/exp_d.json"))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for eps in [0.005, 0.02, 0.05]:
    rows = [r for r in db["rows"] if r["eps"] == eps]
    deltas = [r["delta"] for r in rows]
    diffs = [r["mean_diff"] for r in rows]
    std = [r["std"] for r in rows]
    axes[0].errorbar(deltas, diffs, yerr=std, marker="o", ms=3,
                     label=f"$\\varepsilon$={eps} (slope={db['slopes'][str(eps)]:.2f})")
axes[0].set_xlabel(r"metric perturbation $\|\Delta D_1\|$-scale $\delta$")
axes[0].set_ylabel(r"$\|\gamma^*(D_1)-\gamma^*(D_1+\Delta)\|_F$")
axes[0].set_title("Coupling stability (Exp B): linear in $\\delta$\n"
                  f"$\\partial\\log L/\\partial\\log\\varepsilon$ = "
                  f"{db['eps_exponent']:.2f} (theory hyp.: -1)")
axes[0].legend(fontsize=7)

for eps in [0.005, 0.02, 0.05]:
    rows = [r for r in dd["rows"] if r["eps"] == eps]
    eta = [r["eta"] for r in rows]
    sb = [r["mean_bary_sup"] for r in rows]
    axes[1].plot(eta, sb, marker="o", ms=3,
                 label=f"$\\varepsilon$={eps} (slope={dd['slopes'][str(eps)]:.2f})")
axes[1].set_xlabel(r"relative coupling perturbation $\eta$")
axes[1].set_ylabel(r"$\sup_i\,\|\mathcal{T}_{\gamma'}(x_i)-\mathcal{T}_{\gamma}(x_i)\|$")
axes[1].set_title("Barycentric map stability (Exp D)")
axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/exp_bd_stability.png")
print("fig BD saved")

# ---------------------------------------------------------------- Exp C
dc = json.load(open("results/exp_c.json"))
rows = [r for r in dc["rows"] if r["family"] == "aligned"]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
sc = axes[0].scatter([r["spectral_l1"] for r in rows],
                     [r["gw"] for r in rows],
                     c=[r["prec10"] for r in rows], cmap="viridis_r", s=14)
plt.colorbar(sc, ax=axes[0], label="prec@10")
axes[0].set_xlabel(r"spectral distance $\sum_i|\lambda_i(D_1)-\lambda_i(D_2)|$")
axes[0].set_ylabel("GW divergence")
axes[0].set_title("Spectral predictor vs GW cost\n(ρ=%.2f)" %
                  dc["correlations"]["spectral_l1 vs gw"]["rho"])
sc = axes[1].scatter([r["spectral_l1"] for r in rows],
                     [r["prec10"] for r in rows],
                     c=[r["distortion"] for r in rows], cmap="plasma", s=14)
plt.colorbar(sc, ax=axes[1], label="warp strength")
axes[1].set_xlabel(r"spectral distance $\sum_i|\lambda_i(D_1)-\lambda_i(D_2)|$")
axes[1].set_ylabel("correspondence prec@10")
axes[1].set_title("Spectral predictor vs alignability\n(ρ=%.2f)" %
                  dc["correlations"]["spectral_l1 vs (1-prec10)"]["rho"])
sc = axes[2].scatter([r["gw"] for r in rows], [r["prec10"] for r in rows],
                     c=[r["noise"] for r in rows], cmap="cool", s=14)
plt.colorbar(sc, ax=axes[2], label="noise")
axes[2].set_xlabel("GW divergence"); axes[2].set_ylabel("correspondence prec@10")
axes[2].set_title("GW divergence vs alignability\n(ρ=%.2f)" %
                  dc["correlations"]["gw vs (1-prec10)"]["rho"])
fig.tight_layout(); fig.savefig("figures/exp_c_spectral.png")
print("fig C saved")

# ---------------------------------------------------------------- Exp F
try:
    df = json.load(open("results/exp_f.json"))
    epsl = df["config"]["EPS_LIST"]
    gaps = df["config"]["GAPS"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for g in gaps:
        p1 = [df["aggregate"][f"gap={g},eps={e}"]["prec1"] for e in epsl]
        p10 = [df["aggregate"][f"gap={g},eps={e}"]["prec10"] for e in epsl]
        axes[0].plot(epsl, p1, "o-", ms=3, label=f"gap={g}")
        axes[1].plot(epsl, p10, "o-", ms=3, label=f"gap={g}")
    for ax, t in zip(axes, ["prec@1 (fine structure)", "prec@10 (coarse structure)"]):
        ax.set_xscale("log"); ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel(t)
        ax.set_title(f"MoE fragmentation (Exp F): {t}")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/exp_f_moe.png")
    print("fig F saved")
except FileNotFoundError:
    print("exp F not finished; skipping figure")

# ---------------------------------------------------------------- Exp E
try:
    import os
    path_e = "results/exp_e_merged.json"
    if not os.path.exists(path_e):
        path_e = "results/exp_e.json"
    de = json.load(open(path_e))
    res = de["results"]
    tags = sorted(set(r["tag"] for r in res))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for tag in tags:
        rs = sorted([r for r in res if r["tag"] == tag], key=lambda r: r["eps"])
        axes[0].plot([r["eps"] for r in rs], [r["acc_align"] for r in rs],
                     "o-", ms=3, label=tag)
        axes[1].plot([r["egw"] for r in rs], [r["acc_align"] for r in rs],
                     "o", ms=5, label=tag)
    axes[0].set_xscale("log"); axes[0].set_xlabel(r"$\varepsilon$")
    axes[0].set_ylabel("kNN class-transfer accuracy")
    axes[0].set_title(f"Zero-shot alignment, MNIST nets (n={len(res)} configs)")
    axes[0].legend(fontsize=5, ncol=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("EGW divergence"); axes[1].set_ylabel("kNN class-transfer accuracy")
    axes[1].set_title(f"Error-bound hypothesis (MNIST)\n$\\rho$(GW,acc)={de['correlations']['gw_accalign']:.2f}")
    axes[1].legend(fontsize=5)
    fig.tight_layout(); fig.savefig("figures/exp_e_mnist.png")
    print("fig E saved")
except FileNotFoundError:
    print("exp E not finished; skipping figure")

# ---------------------------------------------------------------- Exp I2/I3
try:
    di2 = json.load(open("results/exp_i2.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for seed in [0, 1, 2]:
        rs = sorted([r for r in di2 if r["seed"] == seed], key=lambda r: r["sigma"])
        axes[0].plot([r["snr"] for r in rs], [r["prec10"] for r in rs], "o-",
                     ms=3, label=f"instance {seed}")
    axes[0].set_xlabel(r"noise level $\sigma / h$"); axes[0].set_ylabel("prec@10")
    axes[0].set_title("Within-instance noise sweep (Exp I2):\ninstance hardness dominates, noise mild")
    axes[0].legend(fontsize=7)
    di3 = json.load(open("results/exp_i3.json"))
    for seed in [0, 1, 2]:
        rs = sorted([r for r in di3 if r["seed"] == seed], key=lambda r: r["N"])
        axes[1].plot([r["N"] for r in rs], [r["prec10"] for r in rs], "o--",
                     ms=3, label=f"instance {seed}")
    axes[1].set_xscale("log"); axes[1].set_xlabel("N (fixed warp instance)")
    axes[1].set_ylabel("prec@10")
    axes[1].set_title("Density at fixed instance (Exp I3):\nno collapse — earlier one was instance artifact")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/exp_i_identifiability.png")
    print("fig I saved")
except FileNotFoundError:
    print("exp I not finished; skipping figure")

# ---------------------------------------------------------------- Exp L (CIFAR)
try:
    dl = json.load(open("results/exp_l.json"))
    res = dl["results"]
    tags = sorted(set(r["tag"] for r in res))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for tag in tags:
        rs = sorted([r for r in res if r["tag"] == tag], key=lambda r: r["eps"])
        axes[0].plot([r["eps"] for r in rs], [r["acc_align"] for r in rs],
                     "o-", ms=3, label=tag)
        axes[1].plot([r["egw"] for r in rs], [r["acc_align"] for r in rs],
                     "o", ms=5, label=tag)
    axes[0].set_xscale("log"); axes[0].set_xlabel(r"$\varepsilon$")
    axes[0].set_ylabel("kNN class-transfer accuracy")
    axes[0].set_title("CIFAR-10: ViT / Mixer / CNN (Exp L)")
    axes[0].legend(fontsize=6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("EGW divergence"); axes[1].set_ylabel("kNN class-transfer accuracy")
    axes[1].set_title(f"CIFAR: $\\rho$(GW,acc)={dl['correlations']['gw_accalign']:.2f}\n"
                      "ViT->Mixer violates monotone law (low GW, poor transfer)")
    axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig("figures/exp_l_cifar.png")
    print("fig L saved")
except FileNotFoundError:
    print("exp L not finished; skipping figure")
