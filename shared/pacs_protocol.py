"""The ONE PACS protocol shared by Tasks 2 and 3.

* Sources = photo, art_painting, cartoon; each split 80/20 (stratified, seed 6304).
* Target  = sketch (all images).
Two separate files are written ON PURPOSE:
    shared/splits/pacs_sources_seed6304.json   <- the only file Task 3 training may read
    shared/splits/pacs_sketch_target.json      <- read only by Task 2 training (labels
                                                  stripped) and the final evaluation scripts
Physically separating them makes "no Sketch image in Task 3 training/selection" easy to
audit: grep the Task 3 training code for TARGET_FILE and you will find nothing.

Run once:  python -m shared.pacs_protocol --root data/PACS
"""
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from common.io import load_json, repo_root, save_json
from shared.pacs import SOURCE_DOMAINS, TARGET_DOMAIN, scan_domain

SPLIT_DIR = repo_root() / "shared" / "splits"
SOURCE_FILE = SPLIT_DIR / "pacs_sources_seed6304.json"
TARGET_FILE = SPLIT_DIR / "pacs_sketch_target.json"


EXPECTED = {"photo": 1670, "art_painting": 2048, "cartoon": 2344, "sketch": 3929}   # 9,991 images in total


def _check(domain, n):
    if n != EXPECTED[domain]:
        print(f"WARNING: {domain} has {n} images, expected {EXPECTED[domain]} -- incomplete or different PACS copy?")


def make_splits(root, seed=6304):
    sources = {}
    for d in SOURCE_DOMAINS:
        items = scan_domain(root, d)
        _check(d, len(items))
        labels = [y for _, y in items]
        tr, va = train_test_split(items, test_size=0.2, stratify=labels, random_state=seed)
        sources[d] = {"train": sorted(tr), "val": sorted(va)}
        print(f"{d:13s} train={len(tr):5d} val={len(va):4d}")
    save_json({"seed": seed, "domains": sources}, SOURCE_FILE)
    target = scan_domain(root, TARGET_DOMAIN)
    _check(TARGET_DOMAIN, len(target))
    save_json({"domain": TARGET_DOMAIN, "items": target}, TARGET_FILE)
    print(f"{TARGET_DOMAIN:13s} all={len(target)}")


def load_source_splits():
    splits = load_json(SOURCE_FILE)["domains"]
    for d, s in splits.items():                                       # leakage guard
        assert d != TARGET_DOMAIN and all(TARGET_DOMAIN not in p for p, _ in s["train"] + s["val"])
    return splits


def load_target_items():
    return load_json(TARGET_FILE)["items"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/PACS")
    ap.add_argument("--seed", type=int, default=6304)
    a = ap.parse_args()
    Path(SPLIT_DIR).mkdir(parents=True, exist_ok=True)
    make_splits(a.root, a.seed)
