"""Exp N: end-to-end benchmarks through stitched models (difficulty ladder).

Protocol per (pair, depth, benchmark):
  intactA : source model alone
  intactB : target model alone (ceiling)
  stitched: run A on the input, map its hidden states at layer l (zero-shot GW
            map fitted on token clouds) into B's layer l, continue B's forward.
  shuffled: same but with a row-permuted coupling (correspondence destroyed).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
from gwxalign.core import (euclidean_distance_matrix, normalize_diameter,
                           out_of_sample_map, solve_egw_gpu, unit)
from openck.registry import TEXT_MODELS
from openck.stitch import _decoder_layers, _token_acts

RES = "results/exp_n.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PAIRS = [("pythia-160m->410m", "pythia-160m", "pythia-410m", 0.5),
         ("pythia-160m->410m", "pythia-160m", "pythia-410m", 1.0),
         ("pythia-410m->1.4b", "pythia-410m", "pythia-1.4b", 1.0),
         ("pythia-70m->160m", "pythia-70m", "pythia-160m", 1.0),
         ("gpt2->gpt2-medium", "gpt2", "gpt2-medium", 1.0)]


# ---------------------------------------------------------------- data
def _mc(ctx, choices, label):
    return dict(ctx=str(ctx), choices=[str(c) for c in choices], label=int(label))


def load_lambada(n=400):
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    rng = np.random.default_rng(0)
    sample_texts = [ds[int(i)]["text"] for i in rng.choice(len(ds), 20,
                                                           replace=False)]
    vocab = sorted({w for t in sample_texts for w in t.split()})
    rows = []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        t = ds[int(i)]["text"]
        words = t.split()
        ctx, target = " ".join(words[:-1]), words[-1]
        cand = [w for w in vocab if w != target]
        distr = [cand[k] for k in rng.choice(len(cand), 3, replace=False)]
        rows.append(_mc(ctx, list(distr) + [target], 3))
    return "lambada", rows


def load_piqa(n=400):
    from datasets import load_dataset
    ds = load_dataset("baber/piqa", split="validation")   # test labels are -1
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        r = ds[int(i)]
        if int(r["label"]) not in (0, 1):
            continue
        rows.append(_mc(r["goal"], [r["sol1"], r["sol2"]], r["label"]))
    return "piqa", rows


def load_sciq(n=400):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="test")
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        r = ds[int(i)]
        ch = [r["correct_answer"], r["distractor1"], r["distractor2"],
              r["distractor3"]]
        order = rng.permutation(4)
        rows.append(_mc(r["question"], [ch[k] for k in order],
                        int(np.where(order == 0)[0][0])))
    return "sciq", rows


def load_arc(name, n=400):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", name, split="test")
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        r = ds[int(i)]
        ch = r["choices"]["text"]
        if r["answerKey"] not in r["choices"]["label"]:
            continue
        rows.append(_mc(r["question"], ch,
                        r["choices"]["label"].index(r["answerKey"])))
    return name, rows


def load_hellaswag(n=400):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")   # test labels hidden
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        r = ds[int(i)]
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        rows.append(_mc(ctx, r["endings"], int(r["label"])))
    return "hellaswag", rows


def load_gsm8k_sd(path="/home/emilio/Dokumente/ai/bm/gsm8k_200.jsonl", n=200):
    """Solution discrimination: LL(correct CoT) vs LL(number-corrupted CoT)."""
    import re
    rng = np.random.default_rng(0)
    rows = []
    with open(path) as f:
        items = [json.loads(l) for l in f if l.strip()]
    for it in items[:n]:
        q = it.get("problem") or it.get("question")
        sol = it.get("solution") or it.get("answer")
        if not q or not sol:
            continue
        nums = re.findall(r"\d+\.?\d*", sol)
        if len(nums) < 2:
            continue
        corrupt = sol
        shuffled = list(nums)
        rng.shuffle(shuffled)
        for o, nw in zip(nums, shuffled):
            corrupt = corrupt.replace(o, nw, 1)
        if corrupt == sol:
            continue
        rows.append(_mc("Question: " + q + "\nSolution:", [sol, corrupt], 0))
    return "gsm8k_sd", rows


def load_all():
    benches = []
    for fn in (load_lambada, load_piqa, load_sciq,
               lambda: load_arc("ARC-Easy"), lambda: load_arc("ARC-Challenge"),
               load_hellaswag, load_gsm8k_sd):
        try:
            name, rows = fn()
            if len(rows) >= 20:
                benches.append((name, rows))
                print(f"  bench {name}: {len(rows)} items", flush=True)
            else:
                print(f"  bench {name}: only {len(rows)} usable items — skipped",
                      flush=True)
        except Exception as e:
            print(f"  bench SKIPPED: {type(e).__name__} {str(e)[:120]}", flush=True)
    return benches


# ---------------------------------------------------------------- scoring
def _ll_of_choice(logits, ids, mask, n_ctx, n_tok):
    """Sum log-prob of tokens n_ctx..n_tok for one sequence."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[n_ctx:n_tok]
    return float(logp[n_ctx - 1:n_tok - 1, :].gather(1, tgt.unsqueeze(1)).sum())


@torch.no_grad()
def mc_accuracy(model, tok, recs, injector=None, max_len=512, batch=8):
    """injector: None | fn(enc_ids, mask) -> tensor (B,T,dB) replacing the
    output of the hooked layer."""
    correct = 0
    hook, hnd = None, None
    if injector is not None:
        hook_layer, mapfn, src_model, src_tok, src_layer = injector

        def hook(module, inp, out):
            hsA = src_model(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"],
                            output_hidden_states=True
                            ).hidden_states[src_layer]
            ZA = hsA.float().cpu().numpy().reshape(-1, hsA.shape[-1])
            Zm = mapfn(ZA).reshape(hsA.shape[0], hsA.shape[1], -1)
            return torch.tensor(Zm, dtype=out[0].dtype if isinstance(out, tuple)
                                else out.dtype, device=DEV)
        hnd = hook_layer.register_forward_hook(hook)
    try:
        for nch in sorted(set(len(r["choices"]) for r in recs)):
            sub = [r for r in recs if len(r["choices"]) == nch]
            for i in range(0, len(sub), batch):
                chunk = sub[i:i + batch]
                lls = np.zeros((len(chunk), nch))
                for ci in range(nch):
                    prompts = [r["ctx"] + r["choices"][ci] for r in chunk]
                    ctxs = [r["ctx"] for r in chunk]
                    enc = tok(prompts, return_tensors="pt", truncation=True,
                              max_length=max_len, padding=True).to(DEV)
                    ect = tok(ctxs, return_tensors="pt", truncation=True,
                              max_length=max_len, padding=True).to(DEV)
                    out = model(input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"])
                    for b in range(len(chunk)):
                        n_ctx = int(ect["attention_mask"][b].sum())
                        n_tok = int(enc["attention_mask"][b].sum())
                        if n_tok <= n_ctx:
                            lls[b, ci] = -1e9
                            continue
                        lls[b, ci] = _ll_of_choice(out.logits[b].float(),
                                                   enc["input_ids"][b], None,
                                                   n_ctx, n_tok)
                pred = lls.argmax(1)
                correct += int(sum(pred[b] == chunk[b]["label"]
                                   for b in range(len(chunk))))
    finally:
        if hnd is not None:
            hnd.remove()
    return correct / len(recs)


# ---------------------------------------------------------------- main
def make_batch_map(HA, HB, T, p, k=10):
    """Vectorized adaptive-bandwidth KDE map for ALL tokens at once.
    Same semantics as gwxalign.core.out_of_sample_map per token."""
    HA2 = np.sum(HA**2, axis=1)

    def map_all(X):
        d2 = (np.sum(X**2, axis=1)[:, None] + HA2[None, :]
              - 2.0 * (X @ HA.T))
        np.maximum(d2, 0.0, out=d2)
        sigma2 = 2.0 * np.partition(d2, min(k, d2.shape[1] - 1),
                                    axis=1)[:, min(k, d2.shape[1] - 1)]
        sigma2 = np.maximum(sigma2, 1e-12)
        W = np.exp(-d2 / sigma2[:, None])
        wp = W @ (T / p[:, None])
        s = wp.sum(axis=1, keepdims=True)
        s[s <= 0] = 1.0
        return (wp / s) @ HB
    return map_all


def main():
    benches = load_all()
    rows = []
    if os.path.exists(RES):
        rows = json.load(open(RES))
    done = {(r["pair"], r["depth"], r["bench"]) for r in rows}

    model_cache, maps = {}, {}

    def get(key, hf_id):
        if key not in model_cache:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            m = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=torch.float16).to(DEV).eval()
            model_cache[key] = (tok, m)
        return model_cache[key]

    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    sents = [t for t in wt["text"] if len(t) > 120][:300]

    for name, mA, mB, dep in PAIRS:
        hfA, hfB = TEXT_MODELS[mA][0], TEXT_MODELS[mB][0]
        tokA, modA = get(mA, hfA)
        tokB, modB = get(mB, hfB)
        nL = len(_decoder_layers(modB))
        li = max(1, round(dep * nL))
        ck = (mA, mB, li)
        if ck not in maps:
            t0 = time.time()
            HA = _token_acts(modA, tokA, sents, li)
            HB = _token_acts(modB, tokB, sents, li)
            D1 = normalize_diameter(euclidean_distance_matrix(HA))
            D2 = normalize_diameter(euclidean_distance_matrix(HB))
            T, _ = solve_egw_gpu(D1, D2, eps=0.02, max_iter=200, tol=1e-8)
            rngp = np.random.default_rng(5)
            Ts = T[rngp.permutation(len(T))]
            maps[ck] = (HA.copy(), HB.copy(), T, Ts)
            print(f"[map] {name} l={li} ({time.time()-t0:.0f}s)", flush=True)
        HA, HB, T, Ts = maps[ck]
        p = unit(len(HA))
        map_stitch = make_batch_map(HA, HB, T, p)
        map_shuffle = make_batch_map(HA, HB, Ts, p)
        hook_layer = _decoder_layers(modB)[max(0, li - 1)]
        src_layer = max(0, li - 1)

        for bench, recs in benches:
            if (name, dep, bench) in done:
                continue
            t0 = time.time()
            accA = mc_accuracy(modA, tokA, recs)
            accB = mc_accuracy(modB, tokB, recs)
            accS = mc_accuracy(modB, tokB, recs,
                               injector=(hook_layer, map_stitch, modA, tokA,
                                         src_layer))
            accX = mc_accuracy(modB, tokB, recs,
                               injector=(hook_layer, map_shuffle, modA, tokA,
                                         src_layer))
            rows.append(dict(pair=name, depth=dep, layer=li, bench=bench,
                             n=len(recs), intactA=accA, intactB=accB,
                             stitched=accS, shuffled=accX))
            with open(RES, "w") as f:
                json.dump(rows, f, indent=2, default=float)
            print(f"[{name} d{dep}] {bench}: A={accA:.3f} B={accB:.3f} "
                  f"stitched={accS:.3f} shuf={accX:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("saved", RES)


if __name__ == "__main__":
    main()
