"""Baselines, the test-results table, loss plots and MIDI statistics."""
import glob
import json
import os

import numpy as np

from .midi_io import read_midi_events
from .tokens import PAD, VOCAB


def baseline_scores(train, test, vocab=VOCAB, k=0.1):
    """Unigram and bigram (add-k smoothed) perplexity/accuracy on test. train/test: int arrays [N, L]."""
    train, test = np.asarray(train), np.asarray(test)
    uni = np.bincount(train[train != PAD].ravel(), minlength=vocab).astype(float) + k
    uni /= uni.sum()
    a, b = train[:, :-1].ravel(), train[:, 1:].ravel()
    keep = b != PAD
    big = np.bincount(a[keep] * vocab + b[keep], minlength=vocab * vocab).reshape(vocab, vocab).astype(float) + k
    big /= big.sum(axis=1, keepdims=True)
    a, b = test[:, :-1].ravel(), test[:, 1:].ravel()
    keep = b != PAD
    a, b = a[keep], b[keep]
    return {
        "unigram_ppl": float(np.exp(-np.log(uni[b]).mean())), "unigram_acc": float((b == uni.argmax()).mean()),
        "bigram_ppl": float(np.exp(-np.log(big[a, b]).mean())), "bigram_acc": float((big[a].argmax(axis=1) == b).mean()),
        "n_tokens": int(len(b)),
    }


def results_table(train_t, test_t, out_dir, names):
    """Prints baselines and every run's best-validation checkpoint scored on the test windows."""
    from .train_loop import load_model, score
    b = baseline_scores(train_t.numpy(), test_t.numpy())
    rows = [("Random guess", float(VOCAB), 1.0 / VOCAB),
            ("Unigram", b["unigram_ppl"], b["unigram_acc"]),
            ("Bigram", b["bigram_ppl"], b["bigram_acc"])]
    for name in names:
        path = f"{out_dir}/{name}_best.pt"
        if not os.path.exists(path):
            print("missing", path)
            continue
        loss, acc = score(load_model(path), test_t)
        rows.append((name, float(np.exp(loss)), acc))
    print(f"{'model':26s} {'test perplexity':>15s} {'test accuracy':>14s}")
    for n, p, a in rows:
        print(f"{n:26s} {p:15.2f} {a:14.1%}")
    return rows


def plot_history(out_dir, names, save_to=None, show=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        p = f"{out_dir}/{name}_history.json"
        if not os.path.exists(p):
            continue
        h = json.load(open(p))
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train"] for r in h], label="train")
        ax.plot(ep, [r["val"] for r in h], label="validation")
        best = min(h, key=lambda r: r["val"])["epoch"]
        ax.axvline(best, ls="--", color="gray", label=f"best val (epoch {best})")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax.legend()
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


# ------------------------------------------------------------------ MIDI metrics
def note_list(path):
    ev = [e for e in read_midi_events(path) if e[1] == "on"]
    return [e[2] for e in ev], [e[3] for e in ev]


def midi_metrics(pitches, vels):
    """pitch-class histogram, repeated-note ratio, share of unique 4-note patterns, velocity std."""
    p = np.asarray(pitches)
    if len(p) < 8:
        return None
    pc = np.bincount(p % 12, minlength=12).astype(float)
    pc /= pc.sum()
    repeated = float((p[1:] == p[:-1]).mean())
    grams = list(zip(*(p[i:].tolist() for i in range(4))))
    unique4 = len(set(grams)) / len(grams)
    return pc, repeated, unique4, float(np.std(vels))


def folder_metrics(folder, max_notes=None):
    rows = []
    for f in glob.glob(folder.rstrip("/\\") + "/*.mid*"):
        try:
            pitches, vels = note_list(f)
            if max_notes:
                pitches, vels = pitches[:max_notes], vels[:max_notes]
            r = midi_metrics(pitches, vels)
        except Exception as e:
            print("skipping", f, type(e).__name__, e)
            continue
        if r is not None:
            rows.append(r + (len(pitches),))
    if not rows:
        return None
    return (np.mean([r[0] for r in rows], axis=0), float(np.mean([r[1] for r in rows])),
            float(np.mean([r[2] for r in rows])), float(np.mean([r[3] for r in rows])),
            float(np.mean([r[4] for r in rows])), len(rows))


def midi_report(real_folder, other_folders, max_notes=None):
    """Compares folders of MIDI pieces with the real reference pieces on equal-length excerpts."""
    real = folder_metrics(real_folder, max_notes)
    if real is None:
        raise ValueError("no usable MIDI files in the real folder")
    print(f"Metrics on: {'first %d notes of each piece' % max_notes if max_notes else 'whole pieces'}")
    print(f"{'folder':24s} files avg_notes repeated_note unique_4grams velocity_std pitchclass_L1_to_real")
    print(f"{real_folder:24s} {real[5]:5d} {real[4]:9.0f} {real[1]:13.3f} {real[2]:13.3f} {real[3]:12.1f} {0.0:12.3f}")
    for folder in other_folders:
        r = folder_metrics(folder, max_notes)
        if r is None:
            print(f"{folder:24s} no usable files")
            continue
        l1 = float(np.abs(r[0] - real[0]).sum())
        print(f"{folder:24s} {r[5]:5d} {r[4]:9.0f} {r[1]:13.3f} {r[2]:13.3f} {r[3]:12.1f} {l1:12.3f}")
