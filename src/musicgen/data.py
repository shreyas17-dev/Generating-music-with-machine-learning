"""MAESTRO download / split, tokenising MIDI files into 1024-token windows, loading windows."""
import csv
import glob
import os
import random
import shutil
import urllib.request
import zipfile

from .midi_io import read_midi_events
from .tokens import events_to_tokens

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def download_maestro(root="data"):
    """Downloads and unzips MAESTRO v3 into <root>/maestro-v3.0.0 (about 56 MB). Returns that folder."""
    os.makedirs(root, exist_ok=True)
    folder = os.path.join(root, "maestro-v3.0.0")
    if not os.path.exists(folder):
        zpath = os.path.join(root, "maestro.zip")
        print("downloading MAESTRO ...")
        urllib.request.urlretrieve(MAESTRO_URL, zpath)
        zipfile.ZipFile(zpath).extractall(root)
    return folder


def split_maestro(maestro_dir, out_dir, counts=None, seed=0):
    """Copies a random choice of pieces into out_dir/{train,validation,test}.
    MAESTRO's own split keeps performances of the same piece together, so there is no leakage."""
    counts = counts or {"train": 300, "validation": 40, "test": 40}
    csv_files = glob.glob(os.path.join(maestro_dir, "*.csv"))
    assert csv_files, f"metadata .csv not found inside {maestro_dir}"
    rows = list(csv.DictReader(open(csv_files[0], encoding="utf-8")))
    random.seed(seed)
    for split, n in counts.items():
        chosen = [r for r in rows if r["split"] == split]
        random.shuffle(chosen)
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        for r in chosen[:n]:
            shutil.copy(os.path.join(maestro_dir, r["midi_filename"]), os.path.join(out_dir, split))
        print(split, len(os.listdir(os.path.join(out_dir, split))), "files")


def build_dataset(folder, out_path, length=1024, stride=1024):
    """Tokenize every MIDI file in `folder` and cut it into windows of `length` tokens.
    Every window is distinct music (no duplicated augmented copies) and needs no padding.
    Saves {'windows': [N, length], 'file_id': [N]} to out_path."""
    import torch
    files = sorted(glob.glob(folder.rstrip("/\\") + "/*.mid*"))
    wins, fids, used = [], [], 0
    for k, f in enumerate(files):
        try:
            toks = events_to_tokens(read_midi_events(f))
        except Exception as e:
            print("skipping", os.path.basename(f), type(e).__name__, e)
            continue
        t = torch.tensor(toks, dtype=torch.long)
        if len(t) >= length:
            w = t.unfold(0, length, stride).clone()
            wins.append(w)
            fids.append(torch.full((len(w),), k, dtype=torch.long))
            used += 1
    data = {"windows": torch.cat(wins), "file_id": torch.cat(fids)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(data, out_path)
    n = len(data["windows"])
    print(f"{folder}: {used} files -> {n} windows of {length} tokens ({n * length / 1e6:.1f}M tokens) -> {out_path}")
    return data


def load_windows(path):
    import torch
    return torch.load(path)["windows"]


def subset_files(path, n_files, seed=0):
    """Windows of a random subset of `n_files` files (for the small-data experiment)."""
    import torch
    d = torch.load(path)
    ids = d["file_id"].unique().tolist()
    random.Random(seed).shuffle(ids)
    keep = torch.tensor(ids[:n_files])
    return d["windows"][torch.isin(d["file_id"], keep)]
