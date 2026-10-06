"""Train this model.   python models/<this folder>/train.py   (run from the repo root)

Reads data/processed/{train,val}.pt (made by scripts/prepare_data.py) and writes
checkpoints/<NAME>_best.pt, <NAME>_ep*.pt and <NAME>_history.json.
"""
import argparse

from config import CONFIG, NAME
from musicgen.data import load_windows
from musicgen.train_loop import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()
    tr = load_windows(f"{args.data}/train.pt")
    va = load_windows(f"{args.data}/val.pt")
    train(NAME, tr, va, args.out, **CONFIG)


if __name__ == "__main__":
    main()
