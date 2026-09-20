"""Step B of Task 3 evaluation -- the ONLY Task 3 script that loads Sketch.
Run it once, after every Task 3 decision (methods, settings, study grid, checkpoints)
is frozen and `evaluate_sources.py` has been run.

  python -m task3.evaluate_sketch [--study sam|dan_dg]

Produces: Sketch acc/F1, change vs ERM, per-class changes, dominant confusions, failure
grids, and the Task 2 vs Task 3 comparison (ERM vs target-aware DAN vs target-free DAN-DG).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import apply_overrides, load_config, load_json, resolve, save_json
from common.metrics import classification_metrics
from common.seed import get_device, set_seed
from shared.diagnostics import extract
from shared.pacs import CLASSES
from shared.pacs_protocol import load_target_items
from task2.evaluation.class_analysis import failure_grid, plot_curves, plot_per_class_delta, row_confusions
from task3.evaluate_sources import MAIN, STUDIES, checkpoint_path, load_model


def main(study=None, overrides=()):
    set_seed(6304)
    cfg = apply_overrides(load_config("task3/configs/base.yaml"), list(overrides))
    root, device = resolve(cfg["data_root"]), get_device()
    out = Path(resolve("task3/results/final"))
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    src_name = f"source_side_study_{study}.csv" if study else "source_side.csv"
    src = pd.read_csv(out / src_name)                  # frozen source-side results
    runs = [r for _, r in STUDIES[study]] if study else MAIN
    save_json({"evaluated_at": time.ctime(), "runs": runs}, out / f"frozen_{study or 'main'}.json")

    target = load_target_items()                       # <-- first Sketch access in Task 3
    y_t = np.array([y for _, y in target])
    rows, per_class, preds, cms = [], {}, {}, {}
    for r in runs:
        _, z, y = extract(load_model(checkpoint_path(cfg, r), device), root, target, device)
        m = classification_metrics(y, z.argmax(1), len(CLASSES))
        rows.append({"method": r, "sketch_acc": m["acc"], "sketch_f1": m["macro_f1"]})
        per_class[r], preds[r], cms[r] = m["per_class_acc"], z.argmax(1), m["confusion"]
    df = src.merge(pd.DataFrame(rows), on="method")
    ref = "erm" if not study else runs[1]              # study: compare to the main setting
    df["delta_sketch_acc_vs_" + ref] = df.sketch_acc - df.set_index("method").loc[ref, "sketch_acc"]
    df.to_csv(out / (f"table_study_{study}.csv" if study else "table_main.csv"), index=False)
    print(df.round(3).to_string(index=False))
    if study:
        return

    pc = pd.DataFrame(per_class, index=CLASSES)
    pc.to_csv(out / "per_class_sketch_acc.csv")
    plot_per_class_delta(per_class, "erm", figs / "per_class_delta.png", "Sketch per-class accuracy change vs ERM")
    analysis = {}
    for r in runs[1:]:
        d = pc[r] - pc["erm"]
        for tag, c in [("largest_gain", d.idxmax()), ("largest_drop", d.idxmin())]:
            analysis.setdefault(r, {})[tag] = {"class": c, "delta_pp": float(d[c]),
                                               "confusions_after": row_confusions(cms[r], CLASSES.index(c)),
                                               "confusions_erm": row_confusions(cms["erm"], CLASSES.index(c))}
        worst = d.idxmin()
        flip = np.where((y_t == CLASSES.index(worst)) & (preds["erm"] == y_t) & (preds[r] != y_t))[0]
        failure_grid(root, target, flip, preds["erm"], preds[r], "ERM", r,
                     figs / f"failures_{r}_{worst}.png", f"{r}: '{worst}' correct under ERM, wrong after")
    save_json(analysis, out / "class_analysis.json")
    plot_curves({r: load_json(Path(resolve("task3/results/runs")) / r / "history.json") for r in runs[1:]},
                figs / "training_curves.png")

    # ---- Task 2 vs Task 3: value of unlabeled Sketch data (Research Question 4) ----------
    t2 = Path(resolve("task2/results/final/per_class_target_acc.csv"))
    if t2.exists():
        t2pc = pd.read_csv(t2, index_col=0)
        cmp = pd.DataFrame({"ERM (shared)": pc["erm"], "DAN (Task 2, sees Sketch)": t2pc["dan"],
                            "DAN-DG (Task 3, no Sketch)": pc["dan_dg"], "SAM (Task 3)": pc["sam"]})
        cmp.to_csv(out / "task2_vs_task3_per_class.csv")
        t2main = pd.read_csv(Path(resolve("task2/results/final/table_main.csv"))).set_index("method")
        save_json({"ERM": float(df.set_index("method").loc["erm", "sketch_acc"]),
                   "DAN_task2": float(t2main.loc["dan", "target_acc"]),
                   "DAN_DG_task3": float(df.set_index("method").loc["dan_dg", "sketch_acc"])},
                  out / "task2_vs_task3_overall.json")
        print(cmp.round(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(STUDIES), default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    main(a.study, a.set)
