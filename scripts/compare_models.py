"""Test-set table (perplexity / accuracy) and loss curves for the trained models.

    python scripts/compare_models.py
    python scripts/compare_models.py --midi-report real_ref gen_B gen_A     # compare folders of MIDI pieces

Needs data/processed/train.pt and test.pt (scripts/prepare_data.py, or copy them from Drive).
"""
import argparse
import os

from musicgen.data import load_windows
from musicgen.evaluate import midi_report, plot_history, results_table

NAMES = ["A_paper_style", "B_lstm_improved", "C_gru_improved"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--midi-report", nargs="+", metavar="FOLDER",
                    help="first folder = real reference pieces, the rest = generated pieces")
    ap.add_argument("--max-notes", type=int, default=150)
    a = ap.parse_args()

    if a.midi_report:
        midi_report(a.midi_report[0], a.midi_report[1:], max_notes=a.max_notes)
        return
    tr = load_windows(f"{a.data}/train.pt")
    te = load_windows(f"{a.data}/test.pt")
    results_table(tr, te, a.ckpt_dir, NAMES)
    os.makedirs(a.out, exist_ok=True)
    plot_history(a.ckpt_dir, NAMES, save_to=f"{a.out}/loss_curves.png", show=False)
    print("loss curves ->", f"{a.out}/loss_curves.png")


if __name__ == "__main__":
    main()
