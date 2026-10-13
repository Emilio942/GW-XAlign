"""Exp M: GW-XAlign on open checkpoints (Hugging Face).

External validity test: does the GW <-> transfer-loss law hold for real
pretrained models? Grid: 9 pairs (Pythia size family, GPT-2 family,
cross-family, a real MoE via switch-base-128, ViT-supervised vs DINO),
matched relative depths, EGW alignment + kNN class transfer + baselines
(relative representations, naive OT, random projection) + token-level
stitching (dPPL) for same-tokenizer LM pairs.

Run:  venv/bin/python src/exp_m_open.py  (resumable; caches under data_open/)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from gwxalign.core import (barycentric_map, euclidean_distance_matrix,
                           knn_transfer_accuracy, normalize_diameter,
                           out_of_sample_map, solve_egw_gpu, unit)
from openck.baselines import (baseline_naive_ot, baseline_random_map,
                              baseline_relative_representations)
from openck.extract import (extract_text, extract_vision, load_acts, save_acts)
from openck.registry import (EPS, N_ALIGN, N_OOS, PAIRS, TEXT_MODELS,
                             VISION_MODELS, layer_indices)

RES = "results/exp_m.json"
DEV = "cuda"


# ---------------------------------------------------------------- data
def load_text_data():
    from datasets import load_dataset
    rng = np.random.default_rng(0)
    ds = load_dataset("fancyzhx/ag_news", split="train")
    idx = rng.choice(len(ds), N_ALIGN + N_OOS, replace=False)
    texts = [ds[int(i)]["text"] for i in idx]
    labels = np.array([ds[int(i)]["label"] for i in idx])
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    sents = [t for t in wt["text"] if len(t) > 120][:600]
    return texts[:N_ALIGN], labels[:N_ALIGN], texts[N_ALIGN:], labels[N_ALIGN:], sents


def load_vision_data():
    from torchvision.datasets import CIFAR10
    ds = CIFAR10("data_cifar", train=False, download=True)
    idx = np.random.default_rng(1).choice(len(ds), N_ALIGN + N_OOS, replace=False)
    images = np.stack([np.asarray(ds.data[i]) for i in idx])
    labels = np.array([ds.targets[i] for i in idx])
    return images[:N_ALIGN], labels[:N_ALIGN], images[N_ALIGN:], labels[N_ALIGN:]


# ---------------------------------------------------------------- extraction
def ensure_text_acts(texts, meta):
    for key, (hf_id, mtype) in TEXT_MODELS.items():
        done = meta.get(key, {}).get("done_text", False)
        if done:
            print(f"  [cache] {key}", flush=True)
            continue
        t0 = time.time()
        print(f"  extracting {key} ({hf_id}) ...", flush=True)
        acts = extract_text(key, hf_id, mtype, texts)
        save_acts(key, "text", "mean", acts)
        meta.setdefault(key, {})["n_layers"] = max(acts.keys())
        meta[key]["done_text"] = True
        with open("data_open/meta.json", "w") as f:
            json.dump(meta, f)
        print(f"  {key}: layers={sorted(acts)} ({time.time()-t0:.0f}s)", flush=True)


def ensure_vision_acts(images, meta):
    for key, hf_id in VISION_MODELS.items():
        if meta.get(key, {}).get("done_vision", False):
            print(f"  [cache] {key}", flush=True)
            continue
        t0 = time.time()
        print(f"  extracting {key} ({hf_id}) ...", flush=True)
        acts = extract_vision(key, hf_id, images)
        save_acts(key, "vision", "cls", acts)
        meta.setdefault(key, {})["n_layers"] = max(acts.keys())
        meta[key]["done_vision"] = True
        with open("data_open/meta.json", "w") as f:
            json.dump(meta, f)
        print(f"  {key}: layers={sorted(acts)} ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- alignment eval
def eval_pair(ZA, ZB, yA, yB, tag, rows):
    D1 = normalize_diameter(euclidean_distance_matrix(ZA))
    D2 = normalize_diameter(euclidean_distance_matrix(ZB))
    T, log = solve_egw_gpu(D1, D2, eps=EPS, max_iter=200, tol=1e-8)
    p = unit(len(ZA))
    B = barycentric_map(T, ZB, p)
    acc = knn_transfer_accuracy(B, ZB, yB, yA, k=10)
    rec = dict(tag=tag, eps=EPS, layer_pair=tag.split("|")[1] if "|" in tag else "",
               egw=log["loss"], acc_align=acc,
               acc_relrep=baseline_relative_representations(ZA, ZB, yB, yA),
               acc_naive_ot=baseline_naive_ot(ZA, ZB, yB, yA),
               acc_random=baseline_random_map(ZA, ZB, yB, yA))
    rows.append(rec)
    print(f"  {tag}: EGW={log['loss']:.4f} acc={acc:.3f} "
          f"relrep={rec['acc_relrep']:.3f} naiveOT={rec['acc_naive_ot']:.3f}",
          flush=True)
    return rec


def main():
    os.makedirs("data_open", exist_ok=True)
    meta = {}
    if os.path.exists("data_open/meta.json"):
        meta = json.load(open("data_open/meta.json"))

    tA, yA, tO, yO, sents = load_text_data()
    print(f"text data: {len(tA)} align / {len(tO)} oos / {len(sents)} wiki",
          flush=True)

    ensure_text_acts(tA, meta)
    vA, vyA, vO, vyO = load_vision_data()
    ensure_vision_acts(vA, meta)

    rows = []
    if os.path.exists(RES):
        rows = json.load(open(RES))["rows"]
    done = {(r["tag"], r.get("depth")) for r in rows}

    for name, mA, mB, family in PAIRS:
        vision = family == "vision"
        mod = "vision" if vision else "text"
        la = meta[mA]["layers"] if "layers" in meta[mA] else \
            layer_indices(meta[mA]["n_layers"])
        lb = meta[mB]["layers"] if "layers" in meta[mB] else \
            layer_indices(meta[mB]["n_layers"])
        yA_, yB_ = (vyA, vyO) if vision else (yA, yA)
        ZB_by_layer = {li: load_acts(mB, mod, "cls" if vision else "mean", li)
                       for li in lb}
        for j in range(min(len(la), len(lb))):
            dep = round((j + 1) / min(len(la), len(lb)), 2)
            if (name, dep) in done:
                continue
            ZA = load_acts(mA, mod, "cls" if vision else "mean", la[j])
            ZB = ZB_by_layer[lb[j]]
            t0 = time.time()
            rec = eval_pair(ZA, ZB, yA_, yB_, f"{name}|d{dep}", rows)
            rec.update(depth=dep, family=family, modelA=mA, modelB=mB,
                       layerA=int(la[j]), layerB=int(lb[j]),
                       elapsed=time.time() - t0)
            with open(RES, "w") as f:
                json.dump(dict(rows=rows), f, indent=2, default=float)

    # ---------------------------------------------------------------- stitching
    from openck.stitch import stitch_pair
    stitch_rows = []
    if os.path.exists("results/exp_m_stitch.json"):
        stitch_rows = json.load(open("results/exp_m_stitch.json"))
    sdone = {(r["pair"], r["depth"]) for r in stitch_rows}
    STITCH_PAIRS = [("pythia-70m->160m", "pythia-70m", "pythia-160m", 0.5),
                    ("pythia-160m->410m", "pythia-160m", "pythia-410m", 0.5),
                    ("pythia-410m->1.4b", "pythia-410m", "pythia-1.4b", 0.5),
                    ("gpt2->gpt2-medium", "gpt2", "gpt2-medium", 0.5)]
    for name, mA, mB, dep in STITCH_PAIRS:
        if (name, dep) in sdone:
            continue
        li = layer_indices(meta[mA]["n_layers"])[
            round(dep * len(layer_indices(meta[mA]["n_layers"]))) - 1]
        print(f"[stitch] {name} @layer {li}", flush=True)
        rec = stitch_pair(TEXT_MODELS[mA][0], TEXT_MODELS[mB][0], li,
                          sents[:300], sents[300:])
        rec.update(pair=name, depth=dep, layer=li)
        stitch_rows.append(rec)
        with open("results/exp_m_stitch.json", "w") as f:
            json.dump(stitch_rows, f, indent=2, default=float)
        print(f"  dPPL={rec['delta_ppl']:.3f} "
              f"(target {rec['target_loss']:.3f}, stitched {rec['stitched_loss']:.3f})",
              flush=True)

    # ---------------------------------------------------------------- summary
    from scipy.stats import spearmanr
    eg = [r["egw"] for r in rows]
    out = dict(rows=rows, stitch=stitch_rows, correlations={
        "egw vs acc": spearmanr(eg, [r["acc_align"] for r in rows])._asdict(),
        "egw vs relrep": spearmanr(eg, [r["acc_relrep"] for r in rows])._asdict(),
        "egw vs naiveOT": spearmanr(eg, [r["acc_naive_ot"] for r in rows])._asdict(),
        "GW better than relrep?": {
            "gw_wins": int(sum(r["acc_align"] > r["acc_relrep"] for r in rows)),
            "n": len(rows)}})
    with open(RES, "w") as f:
        json.dump(out, f, indent=2, default=float)
    for k, v in out["correlations"].items():
        if isinstance(v, dict) and "statistic" in v:
            print(f"rho[{k}] = {v['statistic']:+.3f} (p={v['pvalue']:.1e})")
    print("saved", RES)


if __name__ == "__main__":
    main()
