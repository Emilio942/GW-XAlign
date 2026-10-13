# GW-XAlign

**Zero-Shot Cross-Architecture Representation Alignment via Entropic Gromov-Wasserstein Optimal Transport**
*An empirical scoping study — research artifact, not a finished method paper.*

---

## What this is

This repository contains the code, raw results and a full report for an
autonomous empirical investigation of the following question:

> Does the (entropic) Gromov-Wasserstein divergence between the activation
> geometries of two independently trained neural networks predict the loss of
> a zero-shot, label-free alignment via barycentric transport — and under which
> conditions does it fail?

Sixteen experiment suites (A–L) were executed: synthetic metric-measure
spaces with ground-truth correspondences, five MNIST networks (MLP-128,
MLP-16, CNN-64, multiple seeds) and five CIFAR-10 networks (ViT-S,
MLP-Mixer-S, CNN, multiple seeds).

## TL;DR of findings

| Question | Answer |
|---|---|
| Does GW predict transfer loss? | **Yes within architecture families** (ρ up to −0.96 on MNIST), **attenuated across families** (−0.64 on CIFAR) |
| Strongest violation found | ViT→Mixer: low GW divergence, yet chance-level transfer — the target space is intrinsically weakly class-structured (self-kNN ceiling 0.59). **GW is necessary, not sufficient; read it against the target ceiling.** |
| Zero-shot alignment quality | 94–99 % of the kNN ceiling within families; 85 % cross-family (ViT→CNN); a 128→16 dimension compression costs surprisingly little |
| MoE-style fragmentation (RQ1) | Entropic smoothing ε does **not** rescue it and destroys fine structure at large ε. Ordinal-preserving corruption is repaired unsupervised by a **quantile de-offset** (96–100 % recovery); ordinal-destroying corruption is unrepairable from D1 alone |
| Spectral pre-screen (RQ2) | N-normalized eigenvalue-spectrum distance separates healthy vs unrelated/fragmented pairs with **AUC 0.98–1.0** — a valid triage gate, not a fine-grained predictor |
| Dimensionality mismatch (RQ3) | Ambient dimension is nearly irrelevant; the *source's* intrinsic geometric detail is the bottleneck |
| Stability | Coupling and barycentric maps respond linearly to metric/coupling perturbations with small, ε-robust constants |

Full details, including **refuted sub-hypotheses** (ε-rescue of fragmentation,
1/ε Lipschitz constants, ε^−1/2 concentration factor, debiased-EGW gains,
SNR/density collapse laws, per-point error bounds): see
[`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md).

## What this is NOT (honest limitations)

- **Toy scale**: MNIST/CIFAR-10, ≤ 1M-parameter models, N ≤ 1500 point clouds.
- **No baseline comparison** to existing zero-shot stitching methods
  (Relative Representations (Moschella et al., 2023), ZipIt!, CKA-matching).
- The MoE fragmentation is **simulated** (corruption of distance matrices), not
  a trained Sparse-MoE.
- Single-run configurations for most pairs; no confidence intervals.
- No formal theorems — the empirical claims are supported by controlled
  experiments, the rejected theoretical forms are documented as such.

## Related work (pointers)

- Gromov-Wasserstein OT: Peyré, Cuturi & Solomon (2016); Alvarez-Melis &
  Jaakkola (2018)
- Zero-shot stitching: Moschella et al., *Relative Representations* (ICLR 2023);
  Ainsworth et al., *Git Re-Basin* (2023); *ZipIt!* (2023)
- Entropic OT: Chizat et al. (2020); Pham et al. (2020)

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# experiments (GPU strongly recommended; ~6 h total on an RTX 3060)
.venv/bin/python src/exp_a_concentration.py
.venv/bin/python src/exp_bd_stability.py
.venv/bin/python src/exp_c_spectral.py
.venv/bin/python src/exp_e_mnist.py          # trains + caches MNIST models
.venv/bin/python src/exp_f_moe.py
.venv/bin/python src/exp_g_fragfix.py && .venv/bin/python src/exp_g2_deoffset.py
.venv/bin/python src/exp_h_debias.py
.venv/bin/python src/exp_i_identifiability.py
.venv/bin/python src/exp_i2_sigma_sweep.py && .venv/bin/python src/exp_i3_density.py
.venv/bin/python src/exp_j_deledge.py
.venv/bin/python src/exp_k_calibration.py
.venv/bin/python src/exp_l_cifar.py          # trains + caches CIFAR models
.venv/bin/python src/exp_h2_cifar_debias.py && .venv/bin/python src/exp_e2_more_seeds.py

# figures + report
.venv/bin/python src/make_figures.py
```

Raw numbers land in `results/*.json`, figures in `figures/`, the full write-up
in `reports/RESEARCH_REPORT.md`.

## Repository structure

```
src/gwxalign/core.py        metric-measure construction, GPU/CPU EGW solvers,
                            barycentric + KDE out-of-sample pushforward,
                            spectral diagnostics, precision/kNN metrics
src/exp_*.py                experiment suites A–L (resumable, seeded)
results/                    raw experiment outputs (JSON) + logs
figures/                    generated figures
reports/                    final research report
```

## Compute

All experiments ran on a single RTX 3060 (12 GB) in ≈ 6 h wall-clock.
Solver notes: PPA with capped inner Sinkhorn; float32 GPU requires ε ≥ 0.01
(auto float64 fallback below — `exp(−C/ε)` underflows otherwise).
