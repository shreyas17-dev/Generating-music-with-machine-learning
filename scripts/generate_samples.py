"""Make folders of MIDI pieces for the metrics / listening survey (needs data/processed/test.pt).

    python scripts/generate_samples.py --models A B

Creates real_ref/ and gen_<model>_sampled/ , gen_<model>_greedy/ inside --out (default outputs/samples).
Then:  python scripts/compare_models.py --midi-report outputs/samples/real_ref outputs/samples/gen_B_sampled ...
"""
import argparse
import os

from musicgen.data import load_windows
from musicgen.registry import load_by_name
from musicgen.sampling import generate_set, save_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["A", "B"])
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="outputs/samples")
    a = ap.parse_args()
    te = load_windows(f"{a.data}/test.pt")
    save_reference(te, os.path.join(a.out, "real_ref"))
    for name in a.models:
        model = load_by_name(name, a.ckpt_dir)
        generate_set(model, te, os.path.join(a.out, f"gen_{name}_sampled"),
                     mode="sample", temperature=0.9, top_k=50, top_p=0.95)
        generate_set(model, te, os.path.join(a.out, f"gen_{name}_greedy"), mode="argmax")


if __name__ == "__main__":
    main()
