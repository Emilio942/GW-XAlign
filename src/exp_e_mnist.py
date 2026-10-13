"""Exp E: End-to-end zero-shot cross-architecture alignment on MNIST.

Pipeline:
  1. Train architecturally different networks (MLP-128, MLP-16, CNN-64) on MNIST.
  2. Extract penultimate activations on a held-out evaluation set.
  3. EGW-align cloud A -> cloud B (zero-shot: no correspondence labels, no
     parameter updates). Barycentric pushforward maps A into B's space.
  4. Evaluate class transfer (kNN on labels), and correlate GW divergence with
     transfer loss across pairs / eps values (tests the error-bound hypothesis).
  5. Test dimensionality mismatch (128 -> 16) and class-centroid preservation.
"""
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")
from gwxalign.core import (barycentric_map, euclidean_distance_matrix,
                           knn_transfer_accuracy, normalize_diameter,
                           out_of_sample_map, solve_egw_gpu, solve_gw,
                           spectral_distance, unit)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "data"
N_EVAL = 2000
N_ALIGN = 1500
BATCH = 256
EPOCHS = 3
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class MLP128(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                               nn.Linear(256, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def features(self, x):
        return self.f(x)

    def forward(self, x):
        return self.head(self.features(x))


class MLP16(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(),
                               nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Linear(16, 10)

    def features(self, x):
        return self.f(x)

    def forward(self, x):
        return self.head(self.features(x))


class CNN64(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 64), nn.ReLU())
        self.head = nn.Linear(64, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f(x)

    def forward(self, x):
        return self.head(self.features(x))


def get_data():
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True, transform=tf)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True, transform=tf)
    return tr, te


def train(model, tr, name):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    dl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4)
    model.to(DEV).train()
    for ep in range(EPOCHS):
        corr = tot = 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            corr += (out.argmax(1) == y).sum().item()
            tot += y.numel()
        print(f"  [{name}] epoch {ep}: train acc {corr/tot:.4f}", flush=True)
    return model


@torch.no_grad()
def extract(model, te):
    model.to(DEV).eval()
    dl = DataLoader(te, batch_size=512, shuffle=False, num_workers=4)
    Z, Y, P = [], [], []
    for x, y in dl:
        z = model.features(x.to(DEV))
        p = model(x.to(DEV)).argmax(1)
        Z.append(z.cpu().numpy()); Y.append(y.numpy()); P.append(p.cpu().numpy())
    return np.concatenate(Z), np.concatenate(Y), np.concatenate(P)


def align_and_eval(ZA, yA, ZB, yB, eps, tag, results):
    t0 = time.time()
    idxA = np.random.default_rng(1).choice(len(ZA), N_ALIGN, replace=False)
    idxB = np.random.default_rng(2).choice(len(ZB), N_ALIGN, replace=False)
    XA, yal = ZA[idxA], yA[idxA]
    XB, ybl = ZB[idxB], yB[idxB]
    D1 = normalize_diameter(euclidean_distance_matrix(XA))
    D2 = normalize_diameter(euclidean_distance_matrix(XB))
    T, log = solve_egw_gpu(D1, D2, eps=eps, max_iter=250, tol=1e-8)
    p = unit(len(XA))
    B = barycentric_map(T, XB, p)

    # kNN class transfer: does a mapped A point sit among B points of same class?
    acc_align = knn_transfer_accuracy(B, XB, ybl, yal, k=10)
    # class-centroid fidelity: map per-class centroids of A, measure whether
    # nearest B-class centroid matches the class
    classes = np.arange(10)
    centA = np.stack([XA[yal == c].mean(0) for c in classes])
    centB = np.stack([XB[ybl == c].mean(0) for c in classes])
    mapped_cent = np.stack([out_of_sample_map(centA[c], XA, XB, T, p)
                            for c in classes])
    dc = np.linalg.norm(mapped_cent[:, None] - centB[None, :], axis=2)
    # Hungarian matching of centroids
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(dc)
    cent_acc = float(np.mean(yal[r] == ybl[c]))

    # out-of-sample zero-shot transfer on the remaining eval points
    restA = np.setdiff1d(np.arange(len(ZA)), idxA)[:500]
    oos = np.stack([out_of_sample_map(ZA[i], XA, XB, T, p) for i in restA])
    acc_oos = knn_transfer_accuracy(oos, XB, ybl, yA[restA], k=10)

    _, log_gw = solve_gw(D1, D2, num_iter=200)

    rec = dict(tag=tag, eps=eps, egw=log["loss"], gw=log_gw["loss"],
               acc_align=acc_align, acc_oos=acc_oos, cent_acc=cent_acc,
               elapsed=time.time() - t0)
    results.append(rec)
    print(f"  {tag} eps={eps}: EGW={log['loss']:.4f} acc_align={acc_align:.3f} "
          f"acc_oos={acc_oos:.3f} cent_acc={cent_acc:.2f} "
          f"({rec['elapsed']:.0f}s)", flush=True)


def main():
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(DATA, train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA, train=False, download=True, transform=tf)

    specs = {
        "MLP128": (MLP128, 0), "MLP16": (MLP16, 0), "CNN64": (CNN64, 0),
        "MLP128_s1": (MLP128, 1), "CNN64_s1": (CNN64, 1),
    }
    feats = {}
    t0 = time.time()
    import os
    os.makedirs("data/feats", exist_ok=True)
    for name, (cls, seed) in specs.items():
        fpath = f"data/feats/{name}.npz"
        model = cls(seed=seed)
        ckpt = f"data/feats/{name}.pt"
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEV))
            print(f"[{time.time()-t0:.0f}s] loaded {name}", flush=True)
        else:
            print(f"[{time.time()-t0:.0f}s] training {name}", flush=True)
            train(model, tr, name)
            torch.save(model.state_dict(), ckpt)
        Z, Y, P = extract(model, te)
        acc = (P == Y).mean()
        print(f"  {name}: test acc {acc:.4f}, feature dim {Z.shape[1]}", flush=True)
        np.savez_compressed(fpath, Z=Z, Y=Y)
        feats[name] = (Z, Y)

    PAIRS = [("MLP128", "CNN64"), ("MLP128", "MLP16"),
             ("MLP128", "MLP128_s1"), ("CNN64", "CNN64_s1"),
             ("MLP16", "CNN64")]
    EPS_LIST = [0.002, 0.005, 0.01, 0.02, 0.05]
    results = []
    for a, b in PAIRS:
        ZA, yA = feats[a]
        ZB, yB = feats[b]
        # self-kNN upper bound within B
        acc_self = knn_transfer_accuracy(ZB[:1000], ZB[1000:2000],
                                         yB[1000:2000], yB[:1000], k=10)
        print(f"pair {a}->{b}: self-kNN(B)={acc_self:.3f}", flush=True)
        for eps in EPS_LIST:
            align_and_eval(ZA, yA, ZB, yB, eps, f"{a}->{b}", results)

    # correlation between GW divergence and transfer accuracy
    from scipy.stats import spearmanr
    r1 = spearmanr([r["gw"] for r in results], [r["acc_align"] for r in results])
    r2 = spearmanr([r["egw"] for r in results], [r["acc_align"] for r in results])
    r3 = spearmanr([r["egw"] for r in results], [r["acc_oos"] for r in results])
    print(f"rho(GW, acc_align)={r1.statistic:+.3f}  rho(EGW, acc_align)={r2.statistic:+.3f}  rho(EGW, acc_oos)={r3.statistic:+.3f}")

    with open("results/exp_e.json", "w") as f:
        json.dump(dict(results=results,
                       correlations=dict(gw_accalign=float(r1.statistic),
                                         egw_accalign=float(r2.statistic),
                                         egw_accoos=float(r3.statistic))),
                  f, indent=2, default=float)
    print("saved results/exp_e.json")


if __name__ == "__main__":
    main()
