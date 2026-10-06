"""Training loop, scoring, augmentation and checkpoint loading."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RNNLM, get_device
from .tokens import OFF0, ON0, PAD, SHIFT0, VOCAB

dev = get_device()


def transpose(x, shift):
    """Augmentation: move every note_on / note_off token by `shift` semitones."""
    if shift == 0:
        return x
    y = x.clone()
    on = (x >= ON0) & (x < OFF0)
    off = (x >= OFF0) & (x < SHIFT0)
    y[on] = (x[on] + shift).clamp(ON0, OFF0 - 1)
    y[off] = (x[off] + shift).clamp(OFF0, SHIFT0 - 1)
    return y


def score(model, data, bs=128):
    """(mean next-token cross-entropy, top-1 accuracy). Perplexity = exp(cross-entropy)."""
    model.eval()
    d = next(model.parameters()).device
    nll = ok = n = 0
    with torch.no_grad():
        for i in range(0, len(data), bs):
            b = data[i:i + bs].to(d)
            inp, tar = b[:, :-1], b[:, 1:]
            logits, _ = model(inp)
            m = tar != PAD
            nll += F.cross_entropy(logits[m], tar[m], reduction="sum").item()
            ok += (logits.argmax(-1) == tar)[m].sum().item()
            n += int(m.sum())
    return nll / n, ok / n


def train(name, train_t, val_t, out_dir, cell="lstm", emb=128, hid=256, layers=1, drop=0.1,
          lr=2e-3, weight_decay=0.0, epochs=20, bs=64, augment=False, seed=0):
    """Saves in out_dir: <name>_best.pt (lowest validation loss), <name>_ep5/10/..pt and
    <name>_history.json (train / validation loss every epoch)."""
    os.makedirs(out_dir, exist_ok=True)
    hist_path = f"{out_dir}/{name}_history.json"
    if os.path.exists(hist_path):
        old = json.load(open(hist_path))
        if len(old) >= epochs and os.path.exists(f"{out_dir}/{name}_best.pt"):
            print(f"[{name}] already finished ({len(old)} epochs) - skipping. Delete {hist_path} to retrain.")
            return old
    torch.manual_seed(seed)
    random.seed(seed)
    cfg = dict(cell=cell, emb=emb, hid=hid, layers=layers, drop=drop)
    model = RNNLM(**cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {cell.upper()} {layers} layer(s) x {hid}, dropout {drop}, transposition augmentation={augment}, "
          f"{n_par:,} parameters, {len(train_t)} training windows, device={dev}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist, best = [], float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(train_t))
        tot = cnt = 0
        for i in range(0, len(train_t), bs):
            b = train_t[perm[i:i + bs]].to(dev)
            if augment:
                b = transpose(b, random.randint(-2, 2))
            inp, tar = b[:, :-1], b[:, 1:]
            logits, _ = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tar.reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            cnt += 1
        vl, va = score(model, val_t)
        hist.append(dict(epoch=ep, train=tot / cnt, val=vl, val_acc=va))
        pack = dict(state_dict=model.state_dict(), cfg=cfg, epoch=ep)
        if vl < best:
            best = vl
            torch.save(pack, f"{out_dir}/{name}_best.pt")
        if ep % 5 == 0 or ep == epochs:
            torch.save(pack, f"{out_dir}/{name}_ep{ep}.pt")
        json.dump(hist, open(hist_path, "w"))
        print(f"[{name}] epoch {ep:2d} | train {tot / cnt:.3f} | val {vl:.3f} "
              f"(ppl {np.exp(vl):.1f}, acc {va:.1%}) | {time.time() - t0:.0f}s")
    return hist


def load_model(path, device=None):
    """Loads a checkpoint written by train(). The architecture is stored inside the file."""
    device = device or dev
    ck = torch.load(path, map_location=device)
    model = RNNLM(**ck["cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model
