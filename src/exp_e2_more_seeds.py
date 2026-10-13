"""Exp E2: additional MNIST seeds -> more (GW, accuracy) points for a more
robust correlation estimate. Trains 3 extra models, aligns 5 new pairs,
merges with results/exp_e.json into results/exp_e_merged.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
from exp_e_mnist import (MLP128, MLP16, CNN64, align_and_eval, extract, train,
                         get_data, knn_transfer_accuracy)

DEV = "cuda"


def main():
    tr, te = get_data()
    specs = {"MLP16_s1": (MLP16, 1), "MLP128_s2": (MLP128, 2), "CNN64_s2": (CNN64, 2)}
    feats = {}
    # load previously cached models
    for name in ["MLP128", "MLP16", "CNN64", "MLP128_s1", "CNN64_s1"]:
        d = np.load(f"data/feats/{name}.npz")
        feats[name] = (d["Z"], d["Y"])
    for name, (cls, seed) in specs.items():
        fpath = f"data/feats/{name}.npz"
        if os.path.exists(fpath):
            d = np.load(fpath)
            feats[name] = (d["Z"], d["Y"])
            print(f"loaded {name}", flush=True)
            continue
        model = cls(seed=seed)
        ck = f"data/feats/{name}.pt"
        if not os.path.exists(ck):
            print(f"training {name}", flush=True)
            train(model, tr, name)
            torch.save(model.state_dict(), ck)
        Z, Y, P = extract(model, te)
        print(f"  {name}: test acc {(P==Y).mean():.4f}", flush=True)
        np.savez_compressed(fpath, Z=Z, Y=Y)
        feats[name] = (Z, Y)

    PAIRS = [("MLP128", "MLP16_s1"), ("MLP128", "MLP128_s2"),
             ("MLP16_s1", "CNN64"), ("CNN64", "CNN64_s2"),
             ("MLP128", "CNN64_s2")]
    EPS_LIST = [0.002, 0.005, 0.01, 0.02, 0.05]
    results = []
    for a, b in PAIRS:
        for eps in EPS_LIST:
            align_and_eval(feats[a][0], feats[a][1], feats[b][0], feats[b][1],
                           eps, f"{a}->{b}", results)

    merged = json.load(open("results/exp_e.json"))
    merged["results"] = merged.get("results", []) + results
    from scipy.stats import spearmanr
    rs = merged["results"]
    r1 = spearmanr([r["gw"] for r in rs], [r["acc_align"] for r in rs])
    r2 = spearmanr([r["egw"] for r in rs], [r["acc_align"] for r in rs])
    r3 = spearmanr([r["egw"] for r in rs], [r["acc_oos"] for r in rs])
    merged["correlations"] = dict(gw_accalign=float(r1.statistic),
                                  egw_accalign=float(r2.statistic),
                                  egw_accoos=float(r3.statistic))
    with open("results/exp_e_merged.json", "w") as f:
        json.dump(merged, f, indent=2, default=float)
    print(f"merged rho: GW/acc={r1.statistic:+.3f} EGW/acc={r2.statistic:+.3f} "
          f"EGW/oos={r3.statistic:+.3f}  (n={len(rs)})", flush=True)
    print("saved results/exp_e_merged.json")


if __name__ == "__main__":
    main()
