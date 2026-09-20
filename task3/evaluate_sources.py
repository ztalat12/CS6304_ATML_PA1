"""Step A of Task 3 evaluation -- SOURCE ONLY (safe to run before Sketch is ever loaded).

  python -m task3.evaluate_sources [--study sam|dan_dg]

Per-domain / mean / worst source-validation accuracy + macro-F1, source-domain
separability and the common sharpness proxy for ERM, DAN-DG, SAM (and study runs).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.io import apply_overrides, load_config, resolve, save_json
from common.metrics import classification_metrics
from common.seed import get_device, set_seed
from shared.diagnostics import extract
from shared.models import ResNet18Classifier
from shared.pacs import CLASSES, SOURCE_DOMAINS
from shared.pacs_protocol import load_source_splits
from task3.evaluation.source_diagnostics import fixed_sharpness_batch, sharpness_proxy, source_domain_separability
from task3.methods.erm import erm_checkpoint

RUNS_DIR = "task3/results/runs"
MAIN = ["erm", "dan_dg", "sam"]
STUDIES = {"sam": [("rho=0.01", "sam_rho0.01"), ("rho=0.05", "sam"), ("rho=0.1", "sam_rho0.1")],
           "dan_dg": [("lambda=0.1", "dan_dg_lambda0.1"), ("lambda=1", "dan_dg"), ("lambda=10", "dan_dg_lambda10")]}


def checkpoint_path(cfg, run):
    return erm_checkpoint(cfg) if run == "erm" else Path(resolve(RUNS_DIR)) / run / "best.pt"


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNet18Classifier(num_classes=7, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main(study=None, overrides=()):
    set_seed(6304)
    cfg = apply_overrides(load_config("task3/configs/base.yaml"), list(overrides))
    root, device = resolve(cfg["data_root"]), get_device()
    splits = load_source_splits()
    xb, yb = fixed_sharpness_batch(root, splits, 32, cfg["seed"])
    runs = [r for _, r in STUDIES[study]] if study else MAIN
    rows = []
    for r in runs:
        model = load_model(checkpoint_path(cfg, r), device)
        row = {"method": r}
        for d in SOURCE_DOMAINS:
            _, z, y = extract(model, root, splits[d]["val"], device)
            m = classification_metrics(y, z.argmax(1), len(CLASSES))
            row[f"{d}_acc"], row[f"{d}_f1"] = m["acc"], m["macro_f1"]
        accs = [row[f"{d}_acc"] for d in SOURCE_DOMAINS]
        f1s = [row[f"{d}_f1"] for d in SOURCE_DOMAINS]
        row.update(mean_acc=np.mean(accs), mean_f1=np.mean(f1s), worst_acc=np.min(accs), worst_f1=np.min(f1s))
        row["src_domain_sep"] = source_domain_separability(model, root, splits, device, cfg["seed"])
        row["sharpness"], row["val_batch_loss"], row["grad_norm"] = sharpness_proxy(model, xb, yb, device)
        rows.append(row)
    df = pd.DataFrame(rows)
    if study:
        df.insert(0, "setting", [s for s, _ in STUDIES[study]])
    out = Path(resolve("task3/results/final"))
    out.mkdir(parents=True, exist_ok=True)
    name = f"source_side_study_{study}.csv" if study else "source_side.csv"
    df.to_csv(out / name, index=False)
    # audit trail: the settings Task 3 shares with the Task 2 ERM checkpoint
    save_json({k: cfg.get(k) for k in ["seed", "num_workers", "per_domain_batch", "lr", "weight_decay",
                                       "grad_clip", "max_epochs", "patience", "erm_checkpoint"]},
              out / "protocol.json")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(STUDIES), default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    main(a.study, a.set)
