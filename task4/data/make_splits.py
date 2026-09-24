"""Stratified 90/10 split of the official CIFAR-10 training partition, seed 6304.

  python -m task4.data.make_splits --data_root data            # writes the split file (once)
  python -m task4.data.make_splits --data_root data --verify   # re-derives it and checks it is identical

Handout: "Create a stratified 90/10 split of the official training partition using seed 6304, use only the
training portion for optimization, and select checkpoints by CIFAR-10 validation accuracy."

Stratified = the split is done class by class, so both parts keep CIFAR-10's exact class balance:
5,000 training images per class -> 4,500 train + 500 validation per class -> 45,000 + 5,000 in total.

How: one NumPy random generator seeded with 6304 shuffles the indices of each class in turn (classes in
order 0..9) and the first 10% of each shuffled class become validation. The resulting index lists are saved
as JSON and committed, so the split never depends on library versions again.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common.io import load_json, resolve, save_json
from task4.data.cifar10 import CIFAR10_CLASSES, cifar10

DEFAULT_SPLIT = "task4/data/splits/cifar10_split_seed6304.json"


def make_split(labels, seed=6304, val_fraction=0.1):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in range(int(labels.max()) + 1):
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(val_fraction * len(idx)))
        val += idx[:n_val].tolist()
        train += idx[n_val:].tolist()
    return sorted(train), sorted(val)


def digest(train, val) -> str:
    return hashlib.sha256(json.dumps([train, val]).encode()).hexdigest()


def main(data_root, out, seed, val_fraction, verify, download):
    ds = cifar10(data_root, train=True, transform=None, download=download)
    labels = np.asarray(ds.targets)
    train, val = make_split(labels, seed, val_fraction)
    assert not set(train) & set(val), "train and validation overlap"
    assert len(train) + len(val) == len(labels), "some images were dropped"

    print(f"official training partition: {len(labels)} images")
    print(f"  train {len(train)}   val {len(val)}")
    print(f"  {'class':12s} {'train':>6s} {'val':>5s}")
    for c, name in enumerate(CIFAR10_CLASSES):
        print(f"  {name:12s} {int((labels[train] == c).sum()):6d} {int((labels[val] == c).sum()):5d}")

    out = Path(resolve(out))
    if verify:
        saved = load_json(out)
        same = saved["train"] == train and saved["val"] == val
        print(f"\nverify: the saved split {'IS' if same else 'is NOT'} reproduced exactly from seed {seed}")
        if not same:
            raise SystemExit("Split mismatch - do not continue.")
        return
    if out.exists():
        saved = load_json(out)
        if saved["train"] != train or saved["val"] != val:
            raise SystemExit(f"{out} exists and differs from what seed {seed} gives. Not overwriting.")
        print(f"\n{out} already exists and is identical - nothing to do.")
        return
    save_json({"seed": seed, "val_fraction": val_fraction, "source": "CIFAR-10 official training partition",
               "n_train": len(train), "n_val": len(val), "sha256": digest(train, val),
               "train": train, "val": val}, out)
    print(f"\nwrote {out}  (sha256 {digest(train, val)[:16]}...)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default=DEFAULT_SPLIT)
    ap.add_argument("--seed", type=int, default=6304)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    main(a.data_root, a.out, a.seed, a.val_fraction, a.verify, a.download)
