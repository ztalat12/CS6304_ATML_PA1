"""FINAL Task 2 evaluation. Run ONLY after every checkpoint and setting is frozen.

  python -m task2.evaluate_final                      # main comparison
  python -m task2.evaluate_final --study dan          # or --study dann (controlled study)

This is the first (and only) place where Sketch LABELS are used: target accuracy/F1,
per-class accuracy, confusions and failure cases. Nothing produced here may be fed back
into Task 2 or Task 3 settings.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.io import load_json, resolve, save_json
from common.metrics import classification_metrics
from common.seed import get_device, set_seed
from shared.diagnostics import extract, separability
from shared.models import ResNet18Classifier
from shared.pacs import CLASSES, SOURCE_DOMAINS
from shared.pacs_protocol import load_source_splits, load_target_items
from task2.evaluation.class_analysis import (failure_grid, plot_curves, plot_per_class_delta, plot_study,
                                             row_confusions)

MAIN_RUNS = ["source_only", "dan", "dann", "cdan"]
STUDIES = {"dan": [("lambda=0.1", "dan_lambda0.1"), ("lambda=1", "dan"), ("lambda=10", "dan_lambda10")],
           "dann": [("alpha_max=0.25", "dann_alpha0.25"), ("alpha_max=0.5", "dann_alpha0.5"), ("alpha_max=1", "dann")]}


def load_model(run_dir, device):
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    model = ResNet18Classifier(num_classes=7, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def evaluate_run(run_dir, root, splits, target, device, seed):
    model, ckpt = load_model(run_dir, device)
    out, src_feats = {"best_epoch": ckpt["epoch"]}, []
    for d in SOURCE_DOMAINS:
        f, z, y = extract(model, root, splits[d]["val"], device)
        m = classification_metrics(y, z.argmax(1), len(CLASSES))
        out[f"{d}_acc"], out[f"{d}_f1"] = m["acc"], m["macro_f1"]
        src_feats.append(f)
    out["mean_src_acc"] = float(np.mean([out[f"{d}_acc"] for d in SOURCE_DOMAINS]))
    out["mean_src_f1"] = float(np.mean([out[f"{d}_f1"] for d in SOURCE_DOMAINS]))
    ft, zt, yt = extract(model, root, target, device)
    mt = classification_metrics(yt, zt.argmax(1), len(CLASSES))
    out.update(target_acc=mt["acc"], target_f1=mt["macro_f1"])
    # equal numbers of source-val and target features (subsampled inside `separability`)
    out["domain_sep"] = separability([np.concatenate(src_feats), ft], seed)
    return out, mt, zt.argmax(1)


def main(study=None):
    set_seed(6304)
    device = get_device()
    cfg = load_json(Path(resolve(os.environ.get("RUNS_DIR", "task2/results/runs"))) / "source_only" / "config_resolved.json")
    root, seed = resolve(cfg["data_root"]), cfg["seed"]
    runs_dir = Path(resolve(os.environ.get("RUNS_DIR", "task2/results/runs")))
    out_dir = Path(resolve(os.environ.get("OUT_DIR", "task2/results/final")))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    splits, target = load_source_splits(), load_target_items()

    runs = [r for _, r in STUDIES[study]] if study else MAIN_RUNS
    # record exactly which checkpoints were frozen when target labels were first used
    save_json({"evaluated_at": time.ctime(), "runs": {r: load_json(runs_dir / r / "summary.json")["best_mean_macro_f1"]
                                                    for r in runs}}, out_dir / f"frozen_{study or 'main'}.json")

    rows, per_class, preds, cms = [], {}, {}, {}
    for r in runs:
        res, mt, p = evaluate_run(runs_dir / r, root, splits, target, device, seed)
        rows.append({"method": r, **res})
        per_class[r], preds[r], cms[r] = mt["per_class_acc"], p, mt["confusion"]
    df = pd.DataFrame(rows)

    if study:
        df.insert(0, "setting", [s for s, _ in STUDIES[study]])
        df.to_csv(out_dir / f"table_study_{study}.csv", index=False)
        print(df.round(2).to_string(index=False))
        plot_study(df, fig_dir / f"study_{study}.png", f"Controlled study ({study}): increasing alignment pressure")
        plot_curves({s: load_json(runs_dir / r / "history.json") for s, r in STUDIES[study]},
                    fig_dir / f"training_curves_study_{study}.png")
        return

    base = df.set_index("method").loc["source_only"]
    df["delta_target_acc"] = df.target_acc - base.target_acc
    df.to_csv(out_dir / "table_main.csv", index=False)
    print(df.round(2).to_string(index=False))

    # ---- per-class changes + dominant confusions -------------------------------------
    pc = pd.DataFrame(per_class, index=CLASSES)
    pc.to_csv(out_dir / "per_class_target_acc.csv")
    save_json({"source_only": per_class["source_only"], "classes": CLASSES}, out_dir / "per_class_source_only.json")
    plot_per_class_delta(per_class, "source_only", fig_dir / "per_class_delta.png",
                         "Sketch per-class accuracy change vs Source-only")
    analysis = {"source_only_confusions": {c: row_confusions(cms["source_only"], i) for i, c in enumerate(CLASSES)}}
    y_t = np.array([y for _, y in target])
    # RQ1: which classes drive Source-only's failures? Show some of its mistakes on its weakest class.
    so_worst = CLASSES[int(np.nanargmin(per_class["source_only"]))]
    analysis["source_only_weakest_class"] = so_worst
    wrong = np.where((y_t == CLASSES.index(so_worst)) & (preds["source_only"] != y_t))[0]
    failure_grid(root, target, wrong, preds["source_only"], preds["source_only"], "src-only", "src-only",
                 fig_dir / f"failures_source_only_{so_worst}.png", f"Source-only mistakes on '{so_worst}'")
    for r in runs[1:]:
        delta = pc[r] - pc["source_only"]
        best_c, worst_c = delta.idxmax(), delta.idxmin()
        analysis[r] = {"largest_gain": {"class": best_c, "delta_pp": float(delta[best_c]),
                                        "confusions": row_confusions(cms[r], CLASSES.index(best_c))},
                       "largest_drop": {"class": worst_c, "delta_pp": float(delta[worst_c]),
                                        "confusions_after": row_confusions(cms[r], CLASSES.index(worst_c)),
                                        "confusions_before": row_confusions(cms["source_only"], CLASSES.index(worst_c))}}
        # negative-transfer examples: right before adaptation, wrong after
        flip = np.where((y_t == CLASSES.index(worst_c)) & (preds["source_only"] == y_t) & (preds[r] != y_t))[0]
        failure_grid(root, target, flip, preds["source_only"], preds[r], "src-only", r,
                     fig_dir / f"failures_{r}_{worst_c}.png", f"{r}: '{worst_c}' correct before, wrong after")
    save_json(analysis, out_dir / "class_analysis.json")

    plot_curves({r: load_json(runs_dir / r / "history.json") for r in runs}, fig_dir / "training_curves.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(STUDIES), default=None)
    main(ap.parse_args().study)
