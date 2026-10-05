"""Download MAESTRO, pick 300/40/40 pieces and build the token windows.

    python scripts/prepare_data.py

Output: data/processed/train.pt, val.pt, test.pt
(Not needed for inference. Needed only to retrain or to run scripts/compare_models.py.)
"""
import argparse
import os

from musicgen.data import build_dataset, download_maestro, split_maestro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    args = ap.parse_args()

    maestro = download_maestro(args.root)
    small = os.path.join(args.root, "maestro_small")
    split_maestro(maestro, small)
    proc = os.path.join(args.root, "processed")
    for split, name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        out = os.path.join(proc, f"{name}.pt")
        if not os.path.exists(out):
            build_dataset(os.path.join(small, split), out)


if __name__ == "__main__":
    main()
