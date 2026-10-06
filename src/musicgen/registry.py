"""Maps the short model names "A" / "B" / "C" to their checkpoint files."""
import os

from .train_loop import load_model

CHECKPOINTS = {
    "A": "A_paper_style_best.pt",
    "B": "B_lstm_improved_best.pt",
    "C": "C_gru_improved_best.pt",
}


def checkpoint_path(name, ckpt_dir="checkpoints"):
    key = str(name).upper()
    if key not in CHECKPOINTS:
        raise ValueError(f"unknown model {name!r}; choose one of {sorted(CHECKPOINTS)}")
    path = os.path.join(ckpt_dir, CHECKPOINTS[key])
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}\n"
                                f"Download {CHECKPOINTS[key]} from the shared Drive into {ckpt_dir}/")
    return path


def load_by_name(name, ckpt_dir="checkpoints", device=None):
    return load_model(checkpoint_path(name, ckpt_dir), device=device)
