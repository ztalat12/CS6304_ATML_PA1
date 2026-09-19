"""Step 0 of Task 1: fixed splits.

1. Stratified 80/20 train/validation split of the OFFICIAL training partition
   (seed 6304). The linear heads are trained on 80 % and early-stopped on 20 %.
2. A class-balanced 500-image subset of the OFFICIAL test partition (seed 6304).
   All interventions are evaluated on this subset only. We save the image ids so the
   exact same subset can be reproduced (and so the grader can check it).

Run:  python -m task1.data.make_subset --config task1/configs/task1.yaml
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from common.io import load_config, resolve, save_json
from task1.data.datasets import load_stl10_arrays, to_common_canvas, canvas_to_uint8
import torch


def main(cfg):
    seed, root = cfg["seed"], resolve(cfg["data_root"])
    out_dir = Path(resolve(cfg["splits_dir"]))

    # ---- (1) train/val split -------------------------------------------------
    _, y_train, classes = load_stl10_arrays(root, "train")
    idx = np.arange(len(y_train))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=seed)
    save_json({"seed": seed, "train": np.sort(tr_idx), "val": np.sort(va_idx)},
              out_dir / f"stl10_trainval_seed{seed}.json")

    # ---- (2) class-balanced test subset --------------------------------------
    x_test, y_test, _ = load_stl10_arrays(root, "test")
    k = len(classes)
    per_class = cfg["subset_size"] // k
    rng = np.random.RandomState(seed)
    chosen, counts = [], {}
    for c in range(k):
        pool = np.where(y_test == c)[0]
        take = min(per_class, len(pool))       # if a class is short, use all and DOCUMENT it
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
        counts[classes[c]] = take
    chosen = np.sort(np.asarray(chosen))
    save_json({"seed": seed, "test_indices": chosen, "labels": y_test[chosen],
               "per_class_counts": counts, "classes": classes,
               "imbalanced": len(set(counts.values())) > 1},
              out_dir / f"stl10_test_subset{cfg['subset_size']}_seed{seed}.json")

    # Cache the shared 224x224 canvases once so every later script uses identical pixels.
    canvases = canvas_to_uint8(to_common_canvas(x_test[chosen], cfg["image_size"]))
    cache = Path(resolve(cfg["results_dir"])) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    torch.save({"images": canvases, "labels": torch.as_tensor(y_test[chosen]),
                "test_indices": torch.as_tensor(chosen), "classes": classes},
               cache / "clean_subset.pt")
    print(f"train={len(tr_idx)} val={len(va_idx)} test-subset={len(chosen)} per-class={counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1/configs/task1.yaml")
    main(load_config(ap.parse_args().config))
