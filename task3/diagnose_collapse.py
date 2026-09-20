"""What exactly collapsed in DAN-DG, and why?  SOURCE-ONLY diagnostics -- never loads Sketch.

  python -m task3.diagnose_collapse [--set data_root=...]

Runs on every Task 3 checkpoint that exists (ERM, SAM, the three lambda_DG settings and, if
trained, the unbiased-MMD diagnostic). Everything is measured on the three source VALIDATION
sets, so it is legal before, during or after the Sketch evaluation.

Columns of task3/results/final/collapse_diagnostics.csv
-------------------------------------------------------
mean_f1           mean source-val macro-F1: the number checkpoints are selected on.
top_pred, share   the class the model predicts most often, and the % of all validation images
                  that get it. A collapsed model gives one class to (almost) every image;
                  predicting "person" for everything gives exactly mean macro-F1 = 5.07.
feat_norm         average L2 norm of the 512-d feature F(x). Tests the "shrinking" explanation:
                  the biased MMD's gradient pulls every feature towards a smaller norm
                  (shared/mmd.py). Compare with ERM, which never saw an MMD term.
class_probe       7-way logistic regression trained on FROZEN features (stratified 70/30 split,
                  seed 6304, standardised, C = 1, balanced class weights), scored by balanced
                  accuracy, so chance = 14.3 %. It is the same recipe as the domain probe below
                  but predicting the CLASS. It tells you WHERE the collapse happened:
                    high (say > 80 %)  -> the backbone features still separate the classes;
                                          the head never learned to read them (optimisation).
                    near 14 %          -> the backbone itself erased the class information.
domain_sep        the handout's 3-way source-domain separability (chance 33.3 %).
mmd_biased_8v8    pairwise-source MMD^2 on validation features, 8 vs 8 samples, averaged over
                  200 random draws per pair: the same estimator and sample size as training.
mmd_unbiased_8v8  the same draws scored with the unbiased estimator (0 = no detectable difference;
                  it can be slightly negative).
floor_biased_8v8  the biased estimator on two DISJOINT halves of the SAME domain, i.e. two samples
                  from one distribution. It is the value "perfect alignment" would give, so
                  mmd_biased_8v8 - floor_biased_8v8 is the part that is real domain difference.
ckpt_epoch        the epoch the checkpoint comes from.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.io import apply_overrides, load_config, resolve
from common.metrics import classification_metrics
from common.seed import get_device, set_seed
from shared.diagnostics import extract, separability
from shared.mmd import mmd2
from shared.models import ResNet18Classifier
from shared.pacs import CLASSES, SOURCE_DOMAINS
from shared.pacs_protocol import load_source_splits
from task3.evaluate_sources import checkpoint_path

RUNS = ["erm", "sam", "dan_dg_lambda0.1", "dan_dg", "dan_dg_lambda10", "diag_dan_dg_unbiased"]
PAIRS = [(0, 1), (0, 2), (1, 2)]          # photo-art, photo-cartoon, art-cartoon (as in training)
DRAWS = 200                               # random 8-vs-8 draws per pair


def class_probe(feats, labels, seed=6304):
    """Can a linear classifier still read the 7 classes off the frozen features?"""
    X, y = np.concatenate(feats), np.concatenate(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    probe.fit(Xtr, ytr)
    return float(balanced_accuracy_score(yte, probe.predict(Xte)) * 100.0)


def small_sample_mmd(feats, m=8, draws=DRAWS, seed=6304):
    """(biased, unbiased, biased same-domain floor), each averaged over random m-vs-m draws."""
    rng = np.random.RandomState(seed)
    T = [torch.as_tensor(f, dtype=torch.float32) for f in feats]
    biased, unbiased, floor = [], [], []
    for a, b in PAIRS:
        for _ in range(draws):
            x = T[a][rng.choice(len(T[a]), m, replace=False)]
            y = T[b][rng.choice(len(T[b]), m, replace=False)]
            biased.append(mmd2(x, y).item())
            unbiased.append(mmd2(x, y, unbiased=True).item())
    for d in range(len(T)):
        for _ in range(draws):
            idx = rng.choice(len(T[d]), 2 * m, replace=False)       # two disjoint halves of ONE domain
            floor.append(mmd2(T[d][idx[:m]], T[d][idx[m:]]).item())
    return float(np.mean(biased)), float(np.mean(unbiased)), float(np.mean(floor))


def main(overrides=()):
    set_seed(6304)
    cfg = apply_overrides(load_config("task3/configs/base.yaml"), list(overrides))
    root, device = resolve(cfg["data_root"]), get_device()
    splits = load_source_splits()                 # source splits only -- Sketch is never opened
    rows = []
    for run in RUNS:
        try:
            path = Path(checkpoint_path(cfg, run))
        except FileNotFoundError:
            path = None
        if path is None or not path.exists():
            print(f"  {run:22s} no checkpoint -- skipped")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = ResNet18Classifier(num_classes=len(CLASSES), pretrained=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()

        feats, labels, preds, f1s = [], [], [], []
        for d in SOURCE_DOMAINS:
            f, z, y = extract(model, root, splits[d]["val"], device)
            feats.append(f); labels.append(y); preds.append(z.argmax(1))
            f1s.append(classification_metrics(y, z.argmax(1), len(CLASSES))["macro_f1"])
        all_pred = np.concatenate(preds)
        counts = np.bincount(all_pred, minlength=len(CLASSES))
        top = int(counts.argmax())
        b, u, floor = small_sample_mmd(feats)
        row = {"run": run, "ckpt_epoch": ckpt.get("epoch"), "mean_f1": float(np.mean(f1s)),
               "top_pred": CLASSES[top], "top_pred_share": float(counts[top] / counts.sum() * 100.0),
               "feat_norm": float(np.linalg.norm(np.concatenate(feats), axis=1).mean()),
               "class_probe": class_probe(feats, labels, cfg["seed"]),
               "domain_sep": separability(feats, cfg["seed"]),
               "mmd_biased_8v8": b, "mmd_unbiased_8v8": u, "floor_biased_8v8": floor}
        rows.append(row)
        print(f"  {run:22s} F1 {row['mean_f1']:6.2f} | predicts '{row['top_pred']}' for "
              f"{row['top_pred_share']:5.1f}% | |F(x)| {row['feat_norm']:7.3f} | class probe "
              f"{row['class_probe']:5.1f}% | domain sep {row['domain_sep']:5.1f}% | MMD 8v8 biased "
              f"{b:.3f} (floor {floor:.3f}) unbiased {u:+.4f}", flush=True)
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(resolve("task3/results/final"))
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "collapse_diagnostics.csv", index=False)
    print(f"\nsaved -> {out / 'collapse_diagnostics.csv'}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    main(ap.parse_args().set)
