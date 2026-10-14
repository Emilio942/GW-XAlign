"""Exp N: end-to-end benchmarks through stitched models (difficulty ladder).

Question: when stitching model A's trunk into model B (zero-shot GW map),
how does task performance degrade as benchmark difficulty increases?
Conditions per (pair, layer, benchmark):
  intactA : source model alone (own space, own head)
  intactB : target model alone (ceiling)
  stitched: A trunk -> GW map -> B head
  shuffled: control — coupling rows permuted (correspondence destroyed)

Benchmarks (log-likelihood multiple choice, lm-eval-harness style),
roughly ordered easy -> hard:
  lambada, piqa, sciq, arc_easy, arc_challenge, hellaswag, gsm8k_sd
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
                           solve_egw_gpu, unit)
from openck.registry import TEXT_MODELS

RES = "results/exp_n.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PAIRS = [("pythia-160m->410m", "pythia-160m", "pythia-410m", 0.5),
         ("pythia-160m->410m", "pythia-160m", "pythia-410m", 1.0),
         ("pythia-410m->1.4b", "pythia-410m", "pythia-1.4b", 1.0),
         ("pythia-70m->160m", "pythia-70m", "pythia-160m", 1.0),
         ("gpt2->gpt2-medium", "gpt2", "gpt2-medium", 1.0)]

# ---------------------------------------------------------------- data
def _mc(ctx, choices, label):
    return dict(ctx=ctx, choices=[str(c) for c in choices], label=int(label))


def load_lambada(n=400):
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        t = ds[int(i)]["text"]
        words = t.split()
        ctx, target = " ".join(words[:-1]), words[-1]
        distr = [w for w in set(" ".join(rng.choice(ds["text"], 8).split())
                                .split()) if w != target][:3]
        while len(distr) < 3:
            distr.append("word")
        rows.append(_mc(ctx, distr + [target], 3))
    return "lambada", rows


def load_piqa(n=400):
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split="test", trust_remote_code=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        r = ds[int(i)]
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
        label = int(np.where(order == 0)[0][0])
        rows.append(_mc(r["question"], [ch[k] for k in order], label))
    return "sciq", rows


def load_arc(name, n=400):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", name, split="test")
    rng = np.random.default_rng(0)
    rows = []
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    for i in idx:
        r = ds[int(i)]
        ch = r["choices"]["text"]
        lab = r["choices"]["label"].index(r["answerKey"])
        rows.append(_mc(r["question"], ch, lab))
    return name, rows


def load_hellaswag(n=400):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="test")
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        r = ds[int(i)]
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        rows.append(_mc(ctx, r["endings"], int(r["label"])))
    return "hellaswag", rows


def load_gsm8k_sd(path="/home/emilio/Dokumente/ai/bm/gsm8k_200.jsonl", n=200):
    """Solution-discrimination: LL of the correct CoT solution vs a corrupted
    variant (numbers permuted). Chance = 50%."""
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
        rows.append(_mc("Question: " + q + "\nSolution:", [corrupt, sol], 1))
    return "gsm8k_sd", rows


def load_all():
    benches = []
    for fn in (load_lambada, load_piqa, load_sciq,
               lambda: load_arc("ARC-Easy"), lambda: load_arc("ARC-Challenge"),
               load_hellaswag, load_gsm8k_sd):
        try:
            benches.append(fn())
            print(f"  bench {benches[-1][0]}: {len(benches[-1][1])} items",
                  flush=True)
        except Exception as e:
            print(f"  bench {fn} SKIPPED: {type(e).__name__} {str(e)[:120]}",
                  flush=True)
    return benches


# ---------------------------------------------------------------- scoring
def ll_choice(model, tok, recs, max_len=512, batch=16):
    """Accuracy of multiple-choice via sum-LL of the choice tokens."""
    correct = 0
    with torch.no_grad():
        for i in range(0, len(recs), batch):
            chunk = recs[i:i + batch]
            lls = np.zeros((len(chunk), len(chunk[0]["choices"])))
            for ci in range(len(chunk[0]["choices"])):
                prompts = [r["ctx"] + r["choices"][ci] for r in chunk]
                ctxs = [r["ctx"] for r in chunk]
                enc = tok(prompts, return_tensors="pt", truncation=True,
                          max_length=max_len, padding=True).to(DEV)
                ect = tok(ctxs, return_tensors="pt", truncation=True,
                          max_length=max_len, padding=True).to(DEV)
                out = model(**enc)
                logp = torch.log_softmax(out.logits.float(), dim=-1)
                for b in range(len(chunk)):
                    n_ctx = min(int(ect["attention_mask"][b].sum()),
                                enc["input_ids"].shape[1])
                    ids = enc["input_ids"][b]
                    n_tok = int(enc["attention_mask"][b].sum())
                    if n_tok <= n_ctx:
                        lls[b, ci] = -1e9
                        continue
                    tgt = ids[n_ctx:n_tok]
                    lp = logp[b, n_ctx - 1:n_tok - 1, :].gather(
                        1, tgt.unsqueeze(1)).sum()
                    lls[b, ci] = float(lp)
            pred = lls.argmax(1)
            correct += int(sum(pred[b] == chunk[b]["label"]
                               for b in range(len(chunk))))
    return correct / len(recs)


def _load(hf_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16).to(DEV).eval()
    return tok, m


def main():
    from openck.stitch import _decoder_layers, _token_acts
    benches = load_all()
    rows = []
    if os.path.exists(RES):
        rows = json.load(open(RES))
    done = {(r["pair"], r["depth"], r["bench"]) for r in rows}

    tok_cache, model_cache = {}, {}

    def get(key, hf_id):
        if key not in model_cache:
            model_cache[key] = _load(hf_id)
        return model_cache[key]

    sents = None
    for name, mA, mB, dep in PAIRS:
        hfA, hfB = TEXT_MODELS[mA][0], TEXT_MODELS[mB][0]
        tokA, modA = get(mA, hfA)
        tokB, modB = get(mB, hfB)
        if sents is None:
            from datasets import load_dataset
            wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                              split="test")
            sents = [t for t in wt["text"] if len(t) > 120][:300]

        nL = len(_decoder_layers(modB))
        li = max(1, round(dep * nL))
        ck = (mA, mB, li)
        t0 = time.time()
        if ck not in model_cache.get("maps", {}):
            HA = _token_acts(modA, tokA, sents, li)
            HB = _token_acts(modB, tokB, sents, li)
            D1 = normalize_diameter(euclidean_distance_matrix(HA))
            D2 = normalize_diameter(euclidean_distance_matrix(HB))
            T, _ = solve_egw_gpu(D1, D2, eps=0.02, max_iter=200, tol=1e-8)
            rng = np.random.default_rng(5)
            Ts = T[rng.permutation(len(T))]
            model_cache.setdefault("maps", {})[ck] = (HA.copy(), HB.copy(), T, Ts)
            print(f"[map] {name} l={li} ({time.time()-t0:.0f}s)", flush=True)
        HA, HB, T, Ts = model_cache["maps"][ck]
        p = unit(len(HA))

        from gwxalign.core import out_of_sample_map
        def stitched_map(x):
            return out_of_sample_map(x, HA, HB, T, p)

        def shuffled_map(x):
            return out_of_sample_map(x, HA, HB, Ts, p)

        holder = {}
        layer_list = _decoder_layers(modB)
        hook_layer = layer_list[max(0, li - 1)]

        for bench, recs in benches:
            if (name, dep, bench) in done:
                continue
            accB = ll_choice(modB, tokB, recs)
            accA = ll_choice(modA, tokA, recs)

            def run_with_map(mapfn):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    Z = h.float().cpu().numpy()
                    Zm = np.stack([mapfn(x) for x in Z.reshape(-1, Z.shape[-1])])
                    Zm = Zm.reshape(Z.shape)
                    m = torch.tensor(Zm, dtype=h.dtype, device=DEV)
                    return (m, *out[1:]) if isinstance(out, tuple) else m
                hnd = hook_layer.register_forward_hook(hook)
                try:
                    return ll_choice(modB, tokB, recs)
                finally:
                    hnd.remove()

            accS = run_with_map(stitched_map)
            accX = run_with_map(shuffled_map)
            rows.append(dict(pair=name, depth=dep, layer=li, bench=bench,
                             n=len(recs), intactA=accA, intactB=accB,
                             stitched=accS, shuffled=accX))
            with open(RES, "w") as f:
                json.dump(rows, f, indent=2, default=float)
            print(f"[{name} d{dep}] {bench}: A={accA:.3f} B={accB:.3f} "
                  f"stitched={accS:.3f} shuffled={accX:.3f}", flush=True)

    print("saved", RES)


if __name__ == "__main__":
    main()
