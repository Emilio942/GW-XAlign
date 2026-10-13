"""Activation extraction for open checkpoints (text + vision)."""
from __future__ import annotations

import os

import numpy as np
import torch

CACHE = "data_open"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _load_text(model_id: str, model_type: str):
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16)
    return tok, model.to(DEV).eval()


@torch.no_grad()
def extract_text(model_key: str, hf_id: str, model_type: str,
                 texts: list[str], max_len: int = 128, batch: int = 32,
                 depths=(0.25, 0.5, 0.75, 1.0)) -> dict[int, np.ndarray]:
    """Returns {hidden_states_index: (N, d) float32 mean-pooled activations}."""
    tok, model = _load_text(hf_id, model_type)
    out: dict[int, list] = {}
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                  max_length=max_len, padding=True).to(DEV)
        if model_type == "causal":
            hs = model(**enc, output_hidden_states=True).hidden_states
        else:  # encoder of seq2seq
            hs = model.get_encoder()(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for li in layer_indices(hs.__len__() - 1, depths):
            h = hs[li].float() * mask
            pooled = h.sum(1) / mask.sum(1).clamp(min=1)
            out.setdefault(li, []).append(pooled.cpu().numpy())
    res = {li: np.concatenate(v).astype(np.float32) for li, v in out.items()}
    del model
    torch.cuda.empty_cache()
    return res


def _load_vision(model_id: str):
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    return proc, model.to(DEV).eval()


@torch.no_grad()
def extract_vision(model_key: str, hf_id: str, images: np.ndarray,
                   batch: int = 64, depths=(0.25, 0.5, 0.75, 1.0)) -> dict[int, np.ndarray]:
    """images: uint8 array (N, H, W, 3). Returns {hs_index: (N, d)} CLS-pooled."""
    proc, model = _load_vision(hf_id)
    n_layers = model.config.num_hidden_layers
    out: dict[int, list] = {}
    for i in range(0, len(images), batch):
        px = proc(images=list(images[i:i + batch]), return_tensors="pt").to(DEV)
        hs = model(**px, output_hidden_states=True).hidden_states
        for li in layer_indices(n_layers, depths):
            cls = hs[li][:, 0].float()
            out.setdefault(li, []).append(cls.cpu().numpy())
    res = {li: np.concatenate(v).astype(np.float32) for li, v in out.items()}
    del model
    torch.cuda.empty_cache()
    return res


def save_acts(model_key: str, modality: str, pool: str, acts: dict[int, np.ndarray]):
    d = os.path.join(CACHE, model_key)
    os.makedirs(d, exist_ok=True)
    for li, Z in acts.items():
        np.savez_compressed(os.path.join(d, f"{modality}_{pool}_l{li}.npz"), Z=Z)


def load_acts(model_key: str, modality: str, pool: str, li: int) -> np.ndarray:
    f = os.path.join(CACHE, model_key, f"{modality}_{pool}_l{li}.npz")
    return np.load(f)["Z"].astype(np.float32)
