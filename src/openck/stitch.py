"""Token-level stitching eval for same-tokenizer LM pairs (Pythia/GPT-2 family).

Protocol:
  1. Sample token activations from corpus at layer l in source model A.
  2. Fit EGW coupling on those token clouds (A-space <-> B-space).
  3. Map every token of held-out text A->B via the KDE out-of-sample map.
  4. Forward B normally; then forward B again with hidden state at layer l
     replaced by the mapped tokens (forward hook). Measure loss difference.
"""
from __future__ import annotations

import numpy as np
import torch

from gwxalign.core import (out_of_sample_map, solve_egw_gpu, unit,
                           normalize_diameter, euclidean_distance_matrix)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _decoder_layers(m):
    """Return the decoder layer list for common HF causal LMs."""
    if hasattr(m, "gpt_neox"):
        return m.gpt_neox.h
    if hasattr(m, "transformer"):
        return m.transformer.h
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError("cannot locate decoder layers")


def _token_acts(model, tok, texts, layer_idx, max_len=128, n_max=1500, batch=16):
    """Returns (hidden_states (M,d) float32 for M sampled token positions)."""
    hs_all = []
    with torch.no_grad():
        for i in range(0, min(len(texts), 200), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                      max_length=max_len, padding=True).to(DEV)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer_idx]
            mask = enc["attention_mask"].bool()
            hs_all.append(hs[mask].float().cpu())
    H = torch.cat(hs_all)
    if len(H) > n_max:
        idx = torch.randperm(len(H))[:n_max]
        H = H[idx]
    return H.numpy()


@torch.no_grad()
def stitch_pair(hfA, hfB, layer_idx, texts_align, texts_eval, eps=0.02,
                max_len=128):
    """Returns dict with target/stitched losses on held-out text."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hfA)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mA = AutoModelForCausalLM.from_pretrained(
        hfA, torch_dtype=torch.float16).to(DEV).eval()
    mB = AutoModelForCausalLM.from_pretrained(
        hfB, torch_dtype=torch.float16).to(DEV).eval()

    # 1) token clouds + EGW coupling
    HA = _token_acts(mA, tok, texts_align, layer_idx)
    with torch.no_grad():
        HB = _token_acts(mB, tok, texts_align, layer_idx)
    D1 = normalize_diameter(euclidean_distance_matrix(HA))
    D2 = normalize_diameter(euclidean_distance_matrix(HB))
    T, log = solve_egw_gpu(D1, D2, eps=eps, max_iter=200, tol=1e-8)
    p = unit(len(HA))

    # 2) map function A->B fitted on the token clouds
    def map_tokens(ZA):
        return np.stack([out_of_sample_map(x, HA, HB, T, p) for x in ZA])

    # 3) eval: held-out batches, replace layer output in B
    # hidden_states[li] is the OUTPUT of decoder layer li-1 (hs[0] = embeddings)
    hook_layer = _decoder_layers(mB)[max(0, layer_idx - 1)]
    results = {"target_loss": [], "stitched_loss": []}
    for i in range(0, min(len(texts_eval), 400), 16):
        enc = tok(texts_eval[i:i + 16], return_tensors="pt", truncation=True,
                  max_length=max_len, padding=True).to(DEV)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        res = mB(input_ids=ids, attention_mask=mask, labels=ids)
        results["target_loss"].append(float(res.loss))

        # source activations at layer for the SAME tokens (same tokenizer!)
        with torch.no_grad():
            hsA = mA(input_ids=ids, attention_mask=mask,
                     output_hidden_states=True).hidden_states[layer_idx]
        ZA = hsA.float().cpu().numpy().reshape(-1, hsA.shape[-1])
        ZB_mapped = map_tokens(ZA).reshape(hsA.shape)
        mapped = torch.tensor(ZB_mapped, dtype=hsA.dtype, device=DEV)

        holder = {}

        def hook(module, inp, out):
            if isinstance(out, tuple):
                holder["saved"] = out[0].detach().clone()
                return (mapped, *out[1:])
            holder["saved"] = out.detach().clone()
            return mapped

        handle = hook_layer.register_forward_hook(hook)
        try:
            res2 = mB(input_ids=ids, attention_mask=mask, labels=ids)
        finally:
            handle.remove()
        results["stitched_loss"].append(float(res2.loss))

    out = dict(egw=log["loss"],
               target_loss=float(np.mean(results["target_loss"])),
               stitched_loss=float(np.mean(results["stitched_loss"])))
    out["delta_ppl"] = float(np.exp(out["stitched_loss"]) - np.exp(out["target_loss"]))
    del mA, mB
    torch.cuda.empty_cache()
    return out
