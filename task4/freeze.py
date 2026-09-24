"""Freeze every checkpoint, score definition and threshold BEFORE CIFAR-100 is touched.

  python -m task4.freeze --config task4/configs/base.yaml [--set results_dir=... cache_dir=...]

Uses ONLY CIFAR-10 training and validation outputs (from `extract_outputs --stage known`):
  1. Mahalanobis statistics of every model from its UNAUGMENTED 45k training features
     -> results/frozen/maha_<run>.npz
  2. every unknownness score on the 5,000 validation images, and its threshold
     tau = 95th percentile of validation unknownness                (handout: "accept x when u(x) <= tau")
  3. results/frozen/frozen.json - the freeze record: SHA-256 of each checkpoint, of the score/threshold code and
     of the Mahalanobis statistics; the thresholds; the score definitions; and the pre-registered tables,
     comparisons and failure-selection rule (evaluation/preregistered.py).

The CIFAR-10 TEST set is not read here either: it is evaluation data, like the unknowns.

Re-running this script is safe. If nothing changed, the existing record (and its timestamp) is kept. If
something changed and CIFAR-100 has ALREADY been loaded, the script refuses - a new freeze after seeing the
unknowns must be disclosed - unless you pass --refreeze-reason "...", which archives the old record and access
log under results/frozen/superseded/<time>/ together with your reason.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from common.io import apply_overrides, load_config, load_json, resolve, save_json
from task4 import scores
from task4.audit import code_hashes, now, rel, sha256_arrays, sha256_file
from task4.evaluation import preregistered
from task4.evaluation.metrics import csa, pct_accepted
from task4.evaluation.thresholds import PERCENTILE, fit_threshold
from task4.extract_outputs import FROZEN_FILE, RUNS, load_cached
from task4.scores import mahalanobis as maha


def main(cfg, refreeze_reason=None):
    runs_dir = Path(resolve(cfg["results_dir"]))
    frozen_file = Path(resolve(cfg.get("frozen_file", FROZEN_FILE)))
    frozen_dir = frozen_file.parent
    frozen_dir.mkdir(parents=True, exist_ok=True)
    temperature = float(load_config(resolve("task4/configs/proser.yaml"))["method"]["detection_temperature"])

    rec = {"frozen_at": None, "note": "Everything below was fixed using CIFAR-10 train/val only, before any "
                                      "CIFAR-100 image was loaded.",
           "protocol": {"threshold": f"tau = {PERCENTILE:g}th percentile of validation unknownness; accept iff u <= tau",
                        "percentile": PERCENTILE, "proser_detection_temperature": temperature},
           "score_definitions": scores.DEFINITIONS, "checkpoints": {}, "mahalanobis_stats": {},
           "thresholds": {}, "val_acceptance": {}, "val_csa": {}, "code_sha256": code_hashes(),
           "preregistered": preregistered.as_record()}

    print(f"{'model':8s} {'score':12s} {'tau':>14s} {'val accept':>11s}")
    for run in RUNS:
        ckpt = runs_dir / run / "best.pt"
        sha = sha256_file(ckpt)
        man = load_json(Path(resolve(cfg["cache_dir"])) / run / "manifest.json")
        train = load_cached(cfg["cache_dir"], run, "train", expect_sha=sha)
        val = load_cached(cfg["cache_dir"], run, "val", expect_sha=sha)
        rec["checkpoints"][run] = {"path": rel(ckpt), "sha256": sha, "epoch": man["epoch"], "val_acc": man["val_acc"],
                                   "debug_subset": man.get("debug_subset", 0)}

        stats = maha.fit(train["features"], train["labels"])            # unaugmented 45k training features
        stats_path = frozen_dir / f"maha_{run}.npz"
        maha.save(stats, stats_path)
        rec["mahalanobis_stats"][run] = {"path": rel(stats_path), "sha256": sha256_arrays(stats_path),
                                         "n_train": int(len(train["labels"])),
                                         "var_min": float(stats["var"].min()), "var_max": float(stats["var"].max())}

        u_val = scores.compute(val, stats, temperature)
        rec["thresholds"][run], rec["val_acceptance"][run] = {}, {}
        for s, u in u_val.items():
            tau = fit_threshold(u)
            rec["thresholds"][run][s] = tau
            rec["val_acceptance"][run][s] = pct_accepted(u, tau)
            print(f"{run:8s} {s:12s} {tau:14.6g} {rec['val_acceptance'][run][s]:10.2f}%")
        rec["val_csa"][run] = csa(val["logits"], val["labels"])

    # ---- keep / replace / refuse ----------------------------------------------------------------------------
    log_file = frozen_dir / "unknown_access_log.json"
    if frozen_file.exists():
        old = load_json(frozen_file)
        same = all(old.get(k) == rec[k] for k in ("checkpoints", "code_sha256", "thresholds", "preregistered")) and \
            {r: v["sha256"] for r, v in old["mahalanobis_stats"].items()} == \
            {r: v["sha256"] for r, v in rec["mahalanobis_stats"].items()}
        if same:
            print(f"\nNothing changed since the existing freeze of {old['frozen_at']} - record kept as it is.")
            return old
        if log_file.exists():
            if not refreeze_reason:
                raise SystemExit("REFUSED: the frozen state changed AFTER CIFAR-100 was loaded. Re-freezing now must "
                                 "be disclosed: re-run with --refreeze-reason \"...\" and report it.")
            arch = frozen_dir / "superseded" / now().replace(":", "-")
            arch.mkdir(parents=True)
            shutil.move(str(frozen_file), arch / "frozen.json")
            shutil.move(str(log_file), arch / "unknown_access_log.json")
            (arch / "reason.txt").write_text(refreeze_reason + "\n")
            print(f"\nold record and access log archived in {arch}")
            rec["refrozen_after_unknowns"] = {"reason": refreeze_reason, "archive": rel(arch)}
    elif log_file.exists():
        raise SystemExit(f"{log_file} exists but there is no freeze record - inconsistent state, investigate.")

    rec["frozen_at"] = now()
    save_json(rec, frozen_file)
    print(f"\nFROZEN at {rec['frozen_at']} -> {frozen_file}")
    for run, ck in rec["checkpoints"].items():
        print(f"  {run:8s} epoch {ck['epoch']:3d}  val acc {ck['val_acc']:6.2f}%  sha256 {ck['sha256'][:16]}...")
    print(f"  {len(rec['code_sha256'])} score/threshold/grouping files hashed; Mahalanobis statistics hashed.")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task4/configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--refreeze-reason", default=None)
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set), a.refreeze_reason)
