"""Open-checkpoint experiment (Exp M): model + data registry."""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- models
# text: (hf_id, model_type); model_type in {causal, seq2seq}
TEXT_MODELS = {
    "pythia-70m":   ("EleutherAI/pythia-70m", "causal"),
    "pythia-160m":  ("EleutherAI/pythia-160m", "causal"),
    "pythia-410m":  ("EleutherAI/pythia-410m", "causal"),
    "pythia-1.4b":  ("EleutherAI/pythia-1.4b", "causal"),
    "gpt2":         ("gpt2", "causal"),
    "gpt2-medium":  ("gpt2-medium", "causal"),
    "qwen-0.5b":    ("Qwen/Qwen2.5-0.5B", "causal"),
    "switch-base":  ("google/switch-base-128", "seq2seq"),  # real MoE
}

VISION_MODELS = {
    "vit-sup":  "google/vit-base-patch16-224",   # supervised ImageNet
    "dino-vitb": "facebook/dino-vitb16",          # self-supervised DINO
}

# relative depth of extracted layers (skip pure embedding, include last)
LAYER_DEPTHS = [0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------- pairs
# (name, modelA, modelB, family)
PAIRS = [
    ("pythia-70m->160m",   "pythia-70m",  "pythia-160m", "same-family"),
    ("pythia-160m->410m",  "pythia-160m", "pythia-410m", "same-family"),
    ("pythia-410m->1.4b",  "pythia-410m", "pythia-1.4b", "same-family"),
    ("gpt2->gpt2-medium",  "gpt2",        "gpt2-medium", "same-family"),
    ("qwen->pythia-410m",  "qwen-0.5b",   "pythia-410m", "cross-family"),
    ("gpt2->pythia-160m",  "gpt2",        "pythia-160m", "cross-family"),
    ("switch->pythia-410m","switch-base", "pythia-410m", "cross-family-moe"),
    ("vit-sup->dino",      "vit-sup",     "dino-vitb",   "vision"),
    ("pythia-410m->410m",  "pythia-410m", "pythia-410m", "self"),
]

N_ALIGN = 1500
N_OOS = 500
EPS = 0.02


def layer_indices(n_layers: int, depths=LAYER_DEPTHS) -> list[int]:
    """hidden_states tuple index: 0 = embedding, i = after layer i."""
    return [max(1, min(n_layers, round(r * n_layers))) for r in depths]


def matched_layers(nA: int, nB: int, depths=LAYER_DEPTHS) -> list[tuple[int, int]]:
    """Layer tuples (iA, iB) at matched relative depth."""
    return list(zip(layer_indices(nA, depths), layer_indices(nB, depths)))
