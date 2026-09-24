"""Post-hoc BatchNorm diagnosis for PROSER - CIFAR-10 ONLY (training images + validation images).

  python -m task4.posthoc.bn_diagnosis --set data_root=... [--recal_batches 200]

Question: is PROSER's validation collapse (run 1: 94.6% -> 85-93%, dummy beating every known class on up to 100%
of validation images, while training accuracy stayed at 99.9%) caused by BatchNorm running statistics that were
partly estimated on manifold-mixed features?

Test: evaluate each checkpoint on the 5,000 validation images twice:
  as saved             eval mode with the running statistics accumulated during training
  BN re-estimated      the same weights, running statistics recomputed from 200 batches of REAL training images
and compare, per BatchNorm layer, how far the saved running statistics are from the re-estimated ones.
If the mechanism is right: the final PROSER model recovers after re-estimation, the dummy stops winning on
real images, and the statistics are off mainly in layer3/layer4, the only layers the mixed features pass through.
Vanilla (never trained with mixed features) is the control: re-estimation should change almost nothing.

Only in-memory copies are modified; no checkpoint on disk changes. No CIFAR-100 image is used.
Outputs: task4/results/posthoc/bn_diagnosis.csv, bn_diagnosis_layers.csv, bn_diagnosis.json
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.io import apply_overrides, load_config, resolve, save_json
from common.seed import get_device, set_seed
from task4.data.cifar10 import build_transform, known_split, make_loader
from task4.methods.bn_recalibration import bn_buffers, recalibrate_bn
from task4.models.resnet_cifar import load_model
from task4.train import evaluate

CHECKPOINTS = [("vanilla", "best.pt", "control: never saw mixed features"),
               ("proser", "best.pt", "selected PROSER checkpoint"),
               ("proser", "last.pt", "PROSER after all 50 epochs")]


def stage(name):
    for part in ("layer1", "layer2", "layer3", "layer4"):
        if part in name:
            return part
    return "stem"


def main(cfg, recal_batches):
    cfg["data_root"] = resolve(cfg["data_root"])
    device = get_device()
    set_seed(cfg["seed"])
    cl = bool(cfg.get("channels_last")) and device.type == "cuda"
    runs = Path(resolve(cfg["results_dir"]))
    out = Path(resolve(cfg.get("posthoc_dir", "task4/results/posthoc")))
    out.mkdir(parents=True, exist_ok=True)
    dbg = int(cfg.get("debug_subset") or 0)
    split = resolve(cfg["split_file"])
    val = make_loader(known_split(cfg["data_root"], split, "val", build_transform("eval"), dbg),
                      cfg["eval_batch_size"], False, cfg["num_workers"], cfg["seed"])
    train = make_loader(known_split(cfg["data_root"], split, "train", build_transform("train"), dbg),
                        cfg["batch_size"], True, cfg["num_workers"], cfg["seed"] + 1)

    rows, layer_rows = [], []
    for run, fname, role in CHECKPOINTS:
        path = runs / run / fname
        if not path.exists():
            print(f"skip {path} (not found)")
            continue
        model, ck = load_model(path, device)
        if cl:
            model = model.to(memory_format=torch.channels_last)
        before = evaluate(model, val, device, cl)
        saved = bn_buffers(model)
        n = recalibrate_bn(model, train, device, recal_batches, cl)
        after = evaluate(model, val, device, cl)
        fresh = bn_buffers(model)
        row = {"checkpoint": f"{run}/{fname}", "role": role, "epoch": ck["epoch"],
               "val_acc_as_saved": before["val_acc"], "val_acc_bn_reestimated": after["val_acc"],
               "dummy_wins_as_saved": before.get("val_dummy_wins", np.nan),
               "dummy_wins_bn_reestimated": after.get("val_dummy_wins", np.nan), "recal_images": n}
        per_stage = {}
        for name, (m0, v0) in saved.items():
            m1, v1 = fresh[name]
            # standardised shift of the mean, and log-ratio of the variance, averaged over channels
            dmean = ((m0 - m1).abs() / v1.clamp_min(1e-5).sqrt()).mean().item()
            dvar = (v0.clamp_min(1e-8) / v1.clamp_min(1e-8)).log().abs().mean().item()
            layer_rows.append({"checkpoint": f"{run}/{fname}", "bn_layer": name, "stage": stage(name),
                               "mean_shift_in_sd": dmean, "abs_log_var_ratio": dvar})
            per_stage.setdefault(stage(name), []).append(dmean)
        for st in ("stem", "layer1", "layer2", "layer3", "layer4"):
            row[f"mean_shift_{st}"] = float(np.mean(per_stage.get(st, [np.nan])))
        rows.append(row)

    df, layers = pd.DataFrame(rows), pd.DataFrame(layer_rows)
    df.to_csv(out / "bn_diagnosis.csv", index=False, float_format="%.4f")
    layers.to_csv(out / "bn_diagnosis_layers.csv", index=False, float_format="%.5f")
    save_json({"data": "CIFAR-10 only: running statistics from training images, accuracy on validation images",
               "recal_batches": recal_batches, "rows": df.to_dict("records")}, out / "bn_diagnosis.json")

    pd.set_option("display.width", 220)
    print("Validation accuracy (10 known logits) and % of validation images where the strongest dummy beats every "
          "known class,\nwith the running statistics as saved vs re-estimated from real training images:\n")
    print(df[["checkpoint", "epoch", "val_acc_as_saved", "val_acc_bn_reestimated", "dummy_wins_as_saved",
              "dummy_wins_bn_reestimated"]].round(2).to_string(index=False))
    print("\nHow far the saved running means are from the re-estimated ones (average |shift| in standard deviations),"
          "\nby network stage - mixed features only pass through layer3 and layer4:\n")
    print(df[["checkpoint"] + [f"mean_shift_{s}" for s in ("stem", "layer1", "layer2", "layer3", "layer4")]]
          .round(3).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task4/configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--recal_batches", type=int, default=200)
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set), a.recal_batches)
