"""Exp L (flagship): Cross-architecture zero-shot alignment on CIFAR-10 with
heterogeneous architectures: ViT-S, MLP-Mixer-S, CNN. Tests whether the
rho(GW, transfer-loss) ~ -0.96 law from MNIST MLP/CNN pairs persists across
transformer-scale architectures.

Resumable: checkpoints + features cached under data/cifar/.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from gwxalign.core import (barycentric_map, euclidean_distance_matrix,
                           knn_transfer_accuracy, normalize_diameter,
                           out_of_sample_map, solve_egw_gpu, solve_gw, unit)

DEV = "cuda"
EPOCHS = 20
BATCH = 128
N_ALIGN = 1500
CKPT = "data/cifar"
os.makedirs(CKPT, exist_ok=True)

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class PatchEmbed(nn.Module):
    def __init__(self, patch=4, dim=128, in_ch=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.dim = dim

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)      # (B, T, dim)


class ViTS(nn.Module):
    """Small ViT: dim 128, depth 6, heads 4. Features = CLS token."""

    def __init__(self, seed=0, dim=128, depth=6):
        super().__init__()
        torch.manual_seed(seed)
        self.patch = PatchEmbed(dim=dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, 65, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=256, dropout=0.0,
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def features(self, x):
        B = x.shape[0]
        t = self.patch(x)
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1) + self.pos[:, :t.shape[1] + 1]
        z = self.blocks(t)
        return self.norm(z[:, 0])

    def forward(self, x):
        return self.head(self.features(x))


class MixerS(nn.Module):
    """Small MLP-Mixer: patch 4, dim 128, 8 blocks. Features = mean over tokens."""

    def __init__(self, seed=0, dim=128, depth=8, T=64):
        super().__init__()
        torch.manual_seed(seed)
        self.patch = PatchEmbed(dim=dim)
        self.token_mix = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(T), nn.Linear(T, T)) for _ in range(depth)])
        self.chan_mix = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim)) for _ in range(depth)])
        self.depth, self.T = depth, T
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def trunk(self, x):
        t = self.patch(x)                       # (B,T,D)
        for tm, cm in zip(self.token_mix, self.chan_mix):
            t = t + tm[1](F.gelu(tm[0](t.transpose(1, 2)))).transpose(1, 2)
            t = t + cm[1](F.gelu(cm[0](t)))
        return self.norm(t).mean(1)

    def features(self, x):
        return self.trunk(x)

    def forward(self, x):
        return self.head(self.trunk(x))


class CNNS(nn.Module):
    def __init__(self, seed=0, dim=128):
        super().__init__()
        torch.manual_seed(seed)
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, dim), nn.ReLU())
        self.head = nn.Linear(dim, 10)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.features(x))


def get_data():
    import torchvision
    from torchvision import transforms
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    te_tf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    tr = torchvision.datasets.CIFAR10("data", train=True, download=True,
                                      transform=tr_tf)
    te = torchvision.datasets.CIFAR10("data", train=False, download=True,
                                      transform=te_tf)
    return tr, te


def train(model, tr, name):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    steps = EPOCHS * (50000 // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    dl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=6,
                    persistent_workers=True, drop_last=True)
    scaler = torch.amp.GradScaler("cuda")
    model.to(DEV).train()
    t0 = time.time()
    for ep in range(EPOCHS):
        corr = tot = 0
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            corr += (out.argmax(1) == y).sum().item()
            tot += y.numel()
        print(f"  [{name}] ep {ep+1}/{EPOCHS} acc {corr/tot:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def extract(model, te, mixer_note=False):
    model.to(DEV).eval()
    dl = DataLoader(te, batch_size=512, shuffle=False, num_workers=4)
    Z, Y, P = [], [], []
    for x, y in dl:
        z = model.features(x.to(DEV))
        if isinstance(z, tuple):
            z = z[0]
        p = model(x.to(DEV))
        if isinstance(p, tuple):
            p = p[0]
        Z.append(z.float().cpu().numpy())
        Y.append(y.numpy())
        P.append(p.argmax(1).cpu().numpy())
    return np.concatenate(Z), np.concatenate(Y), np.concatenate(P)


def align_and_eval(ZA, yA, ZB, yB, eps, tag, results):
    t0 = time.time()
    idxA = np.random.default_rng(1).choice(len(ZA), N_ALIGN, replace=False)
    idxB = np.random.default_rng(2).choice(len(ZB), N_ALIGN, replace=False)
    XA, yal, XB, ybl = ZA[idxA], yA[idxA], ZB[idxB], yB[idxB]
    D1 = normalize_diameter(euclidean_distance_matrix(XA))
    D2 = normalize_diameter(euclidean_distance_matrix(XB))
    T, log = solve_egw_gpu(D1, D2, eps=eps, max_iter=250, tol=1e-8)
    p = unit(N_ALIGN)
    B = barycentric_map(T, XB, p)
    acc_align = knn_transfer_accuracy(B, XB, ybl, yal, k=10)
    restA = np.setdiff1d(np.arange(len(ZA)), idxA)[:500]
    oos = np.stack([out_of_sample_map(ZA[i], XA, XB, T, p) for i in restA])
    acc_oos = knn_transfer_accuracy(oos, XB, ybl, yA[restA], k=10)
    _, log_gw = solve_gw(D1, D2, num_iter=200)
    rec = dict(tag=tag, eps=eps, egw=log["loss"], gw=log_gw["loss"],
               acc_align=acc_align, acc_oos=acc_oos,
               elapsed=time.time() - t0)
    results.append(rec)
    with open("results/exp_l.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  {tag} eps={eps}: EGW={log['loss']:.4f} align={acc_align:.3f} "
          f"oos={acc_oos:.3f} ({rec['elapsed']:.0f}s)", flush=True)
    return rec


def main():
    tr, te = get_data()
    specs = {
        "ViT_S": (ViTS, 0), "ViT_S_s1": (ViTS, 1), "Mixer_S": (MixerS, 0),
        "CNN_S": (CNNS, 0), "CNN_S_s1": (CNNS, 1),
    }
    feats = {}
    for name, (cls, seed) in specs.items():
        fpath = f"{CKPT}/{name}.npz"
        if os.path.exists(fpath):
            d = np.load(fpath)
            feats[name] = (d["Z"], d["Y"])
            print(f"loaded features {name} {d['Z'].shape}", flush=True)
            continue
        model = cls(seed=seed)
        ck = f"{CKPT}/{name}.pt"
        if os.path.exists(ck):
            model.load_state_dict(torch.load(ck, map_location=DEV))
            print(f"loaded ckpt {name}", flush=True)
        else:
            print(f"training {name}", flush=True)
            train(model, tr, name)
            torch.save(model.state_dict(), ck)
        Z, Y, P = extract(model, te)
        acc = (P == Y).mean()
        print(f"  {name}: CIFAR test acc {acc:.4f}, dim {Z.shape[1]}", flush=True)
        np.savez_compressed(fpath, Z=Z[:3000], Y=Y[:3000])
        feats[name] = (Z[:3000], Y[:3000])

    # self-kNN ceilings
    for name, (Z, Y) in feats.items():
        c = knn_transfer_accuracy(Z[:1000], Z[1000:2000], Y[1000:2000],
                                  Y[:1000], k=10)
        print(f"self-kNN({name}) = {c:.3f}", flush=True)

    PAIRS = [("ViT_S", "Mixer_S"), ("ViT_S", "CNN_S"), ("Mixer_S", "CNN_S"),
             ("ViT_S", "ViT_S_s1"), ("CNN_S", "CNN_S_s1")]
    EPS_LIST = [0.002, 0.005, 0.01, 0.02, 0.05]
    results = []
    for a, b in PAIRS:
        for eps in EPS_LIST:
            align_and_eval(feats[a][0], feats[a][1], feats[b][0], feats[b][1],
                           eps, f"{a}->{b}", results)

    from scipy.stats import spearmanr
    r1 = spearmanr([r["gw"] for r in results], [r["acc_align"] for r in results])
    r2 = spearmanr([r["egw"] for r in results], [r["acc_align"] for r in results])
    r3 = spearmanr([r["egw"] for r in results], [r["acc_oos"] for r in results])
    print(f"rho(GW, acc_align)={r1.statistic:+.3f} "
          f"rho(EGW, acc_align)={r2.statistic:+.3f} "
          f"rho(EGW, acc_oos)={r3.statistic:+.3f}", flush=True)
    with open("results/exp_l.json", "w") as f:
        json.dump(dict(results=results,
                       correlations=dict(gw_accalign=float(r1.statistic),
                                         egw_accalign=float(r2.statistic),
                                         egw_accoos=float(r3.statistic))),
                  f, indent=2, default=float)
    print("saved results/exp_l.json")


if __name__ == "__main__":
    main()
