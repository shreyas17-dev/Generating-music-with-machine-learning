"""Choosing the next token and continuing a prompt. numpy only at import time (torch is loaded lazily)."""
import os

import numpy as np

from .tokens import PAD


def sample_next(logits, mode="sample", temperature=0.9, top_k=50, top_p=0.95, rng=None):
    """Pick the next token id from a vector of logits.
    mode='argmax'  : always the most likely token (greedy)
    mode='sample'  : temperature, then top-k, then top-p (nucleus) filtering, then random draw"""
    lg = np.asarray(logits, dtype=np.float64).copy()
    lg[PAD] = -np.inf
    if mode == "argmax":
        return int(lg.argmax())
    rng = rng if rng is not None else np.random.default_rng()
    lg = lg / max(temperature, 1e-6)
    if top_k and top_k < len(lg):
        kth = np.partition(lg, -top_k)[-top_k]
        lg[lg < kth] = -np.inf
    p = np.exp(lg - lg.max())
    p /= p.sum()
    if top_p and top_p < 1.0:
        order = np.argsort(-p)
        before = np.cumsum(p[order]) - p[order]       # probability mass ahead of each token
        p[order[before > top_p]] = 0.0
        p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate(model, prompt_tokens, max_new=800, seed=0, **sampling):
    """Continue `prompt_tokens` (list of token ids) by exactly `max_new` tokens (a hard cap, so it always finishes).
    Returns prompt + continuation as one list."""
    import torch
    if len(prompt_tokens) == 0:
        raise ValueError("prompt is empty - the model needs at least one token to continue from")
    rng = np.random.default_rng(seed)
    model.eval()
    d = next(model.parameters()).device
    with torch.no_grad():
        logits, state = model(torch.tensor([list(prompt_tokens)], dtype=torch.long, device=d))
        out = list(prompt_tokens)
        for _ in range(max_new):
            nxt = sample_next(logits[0, -1].float().cpu().numpy(), rng=rng, **sampling)
            out.append(nxt)
            logits, state = model(torch.tensor([[nxt]], dtype=torch.long, device=d), state)
    return out


def _picks(test_t, n):
    import torch
    return torch.linspace(0, len(test_t) - 1, n).long().tolist()


def generate_set(model, test_t, folder, n=10, prompt_len=128, max_new=800, **sampling):
    """n pieces, each continuing the first `prompt_len` tokens of a different test window."""
    from .midi_io import write_midi
    os.makedirs(folder, exist_ok=True)
    for k, w in enumerate(_picks(test_t, n), 1):
        toks = generate(model, test_t[w, :prompt_len].tolist(), max_new=max_new, seed=k, **sampling)
        write_midi(toks, f"{folder}/piece{k}.mid")
    print(folder, "done:", len(os.listdir(folder)), "files")


def save_reference(test_t, folder, n=10):
    """The real test windows the prompts were taken from, written to MIDI the same way."""
    from .midi_io import write_midi
    os.makedirs(folder, exist_ok=True)
    for k, w in enumerate(_picks(test_t, n), 1):
        write_midi(test_t[w].tolist(), f"{folder}/piece{k}.mid")
    print(folder, "done:", len(os.listdir(folder)), "files")
