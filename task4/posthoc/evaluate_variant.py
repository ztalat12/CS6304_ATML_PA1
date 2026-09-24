"""Post-hoc evaluation of the PROSER + BatchNorm-recalibration variant (configs/proser_bncal.yaml).

  python -m task4.posthoc.evaluate_variant --set data_root=... cache_dir=...

This is NOT part of the pre-registered evaluation. The variant was designed after run 1, from CIFAR-10 validation
evidence only (PROSER's validation collapse; see posthoc/bn_diagnosis.py). Run 1's CIFAR-100 results had been
seen by then, so everything here is labelled post-hoc and reported separately from Tables 1 and 2.

The same discipline as the main pipeline, in its own files:
  1. Extract the variant's CIFAR-10 val/test outputs and compute its thresholds (95th percentile of validation
     unknownness) for MLS and the PROSER detection score.
  2. Write results/posthoc/frozen_posthoc.json: SHA-256 of the variant checkpoint and of the post-hoc code, the
     thresholds, and a pointer to the main freeze record (its time and SHA-256). This happens BEFORE the variant
     sees any CIFAR-100 image.
  3. Load CIFAR-100 through the usual guard (which re-verifies the MAIN freeze record and logs the access), then
     extract the variant's unknown outputs.
  4. Compare with the pre-registered models on the same test and unknown images: Vanilla-MLS, PROSER-MLS and
     PROSER-detect from the main cache, next to the variant's MLS and detection rows. Uncertainty uses the same
     tools as Table 2: bootstrap AUROC CIs, paired bootstrap differences and McNemar tests, with Holm correction
     within this post-hoc family.

The main freeze record, main results and main tables are never modified.

Outputs in results/posthoc/:
  frozen_posthoc.json, table_posthoc_proser_bncal.csv, auroc_comparisons_posthoc.csv, rejection_mcnemar_posthoc.csv,
  csa_mcnemar_posthoc.csv, fig_proser_bn.(png|pdf), summary_posthoc.json
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import apply_overrides, load_config, load_json, resolve, save_json
from common.seed import get_device, set_seed
from task4 import scores
from task4.audit import now, rel, sha256_file, sha256_text, verify_frozen
from task4.data.cifar10 import build_transform, known_split
from task4.evaluate_osr import auroc_comparisons, csa_mcnemar, rejection_mcnemar, table
from task4.evaluation.metrics import pct_accepted
from task4.evaluation.stats import bootstrap_aurocs
from task4.evaluation.thresholds import fit_threshold
from task4.extract_outputs import FROZEN_FILE, extract, load_cached
from task4.models.resnet_cifar import load_model

VARIANT = "proser_bncal"
POSTHOC_CODE = ["task4/configs/proser_bncal.yaml", "task4/methods/bn_recalibration.py", "task4/train.py",
                "task4/posthoc/bn_diagnosis.py", "task4/posthoc/evaluate_variant.py"]
ROWS = [("vanilla", "mls"), ("proser", "mls"), ("proser", "proser"), (VARIANT, "mls"), (VARIANT, "proser")]
PAIRS = [((VARIANT, "mls"), ("vanilla", "mls")), ((VARIANT, "proser"), ("vanilla", "mls")),
         ((VARIANT, "mls"), ("proser", "mls")), ((VARIANT, "proser"), ("proser", "proser"))]
CSA_PAIRS = [(VARIANT, "vanilla"), (VARIANT, "proser")]


def save_npz(path, out):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: (v.astype("U") if v.dtype.kind in "OU" else v) for k, v in out.items()})


def main(cfg, B, refreeze_reason=None):
    cfg["data_root"] = resolve(cfg["data_root"])
    main_rec = verify_frozen(cfg.get("frozen_file", FROZEN_FILE))
    main_sha = sha256_file(cfg.get("frozen_file", FROZEN_FILE))
    T = main_rec["protocol"]["proser_detection_temperature"]
    pdir = Path(resolve(cfg.get("posthoc_dir", "task4/results/posthoc")))
    pdir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(resolve(cfg.get("posthoc_runs_dir", "task4/results/posthoc/runs"))) / VARIANT / "best.pt"
    vcache = Path(resolve(cfg["cache_dir"])) / "posthoc" / VARIANT
    device = get_device()
    set_seed(cfg["seed"])
    cl = bool(cfg.get("channels_last")) and device.type == "cuda"
    model, ck = load_model(ckpt, device)
    if cl:
        import torch
        model = model.to(memory_format=torch.channels_last)
    sha = sha256_file(ckpt)
    print(f"variant {VARIANT}: {ckpt} (epoch {ck['epoch']}, val acc {ck['val_acc']:.2f}%, sha256 {sha[:16]}...)")

    # ---- 1. CIFAR-10 outputs and thresholds (no CIFAR-100 yet) --------------------------------------------------
    tf = build_transform("eval")
    dbg = int(cfg.get("debug_subset") or 0)
    man = vcache / "manifest.json"
    if not (man.exists() and load_json(man)["checkpoint_sha256"] == sha and (vcache / "test.npz").exists()):
        for split in ("val", "test"):
            ds = known_split(cfg["data_root"], resolve(cfg["split_file"]), split, tf, dbg)
            out = extract(model, ds, device, cfg["eval_batch_size"], cfg["num_workers"], cl)
            out["index"] = np.asarray(ds.indices)
            save_npz(vcache / f"{split}.npz", out)
        save_json({"checkpoint": str(ckpt), "checkpoint_sha256": sha, "epoch": ck["epoch"]}, man)
    Vv = {s: dict(np.load(vcache / f"{s}.npz")) for s in ("val", "test")}
    acc_val = 100 * np.mean(Vv["val"]["logits"].argmax(1) == Vv["val"]["labels"])
    assert round(abs(acc_val - ck["val_acc"]) * len(Vv["val"]["labels"]) / 100) <= 2, "variant extraction not faithful"
    u_val = scores.compute(Vv["val"], None, T)
    thresholds = {s: fit_threshold(u_val[s]) for s in ("mls", "proser")}
    val_accept = {s: pct_accepted(u_val[s], thresholds[s]) for s in thresholds}

    # ---- 2. post-hoc freeze record (before any CIFAR-100 image reaches the variant) -----------------------------
    pfile = pdir / "frozen_posthoc.json"
    rec = {"frozen_at": None, "label": "POST-HOC - designed after run 1 from CIFAR-10 validation evidence only; not "
                                       "part of the pre-registered evaluation",
           "variant_checkpoint": {"path": rel(ckpt), "sha256": sha, "epoch": ck["epoch"], "val_acc": ck["val_acc"]},
           "thresholds": thresholds, "val_acceptance": val_accept,
           "code_sha256": {p: sha256_text(p) for p in POSTHOC_CODE},
           "main_freeze": {"frozen_at": main_rec["frozen_at"], "sha256": main_sha}}
    unknown_done = (vcache / "unknown.npz").exists()
    if pfile.exists():
        old = load_json(pfile)
        same = all(old.get(k) == rec[k] for k in ("variant_checkpoint", "thresholds", "code_sha256", "main_freeze"))
        if same:
            rec = old
            print(f"post-hoc freeze record unchanged since {old['frozen_at']} - kept")
        elif unknown_done and not refreeze_reason:
            raise SystemExit("REFUSED: the post-hoc state changed after the variant's CIFAR-100 outputs exist. "
                             "Re-run with --refreeze-reason \"...\" and report it.")
        else:
            rec["refrozen"] = refreeze_reason or "changed before any CIFAR-100 access"
    if rec["frozen_at"] is None:
        rec["frozen_at"] = now()
        save_json(rec, pfile)
        print(f"POST-HOC FROZEN at {rec['frozen_at']} -> {pfile}")
    print(f"  thresholds: MLS tau = {thresholds['mls']:.6g}, PROSER-detect tau = {thresholds['proser']:.6g} "
          f"(val acceptance {val_accept['mls']:.2f}% / {val_accept['proser']:.2f}%)")

    # ---- 3. CIFAR-100 unknowns through the guard ----------------------------------------------------------------
    from task4.data.cifar100_unknowns import load_unknowns
    if not (unknown_done and load_json(man)["checkpoint_sha256"] == sha):
        unknowns = load_unknowns(cfg["data_root"], tf, cfg.get("frozen_file", FROZEN_FILE))
        out = extract(model, unknowns, device, cfg["eval_batch_size"], cfg["num_workers"], cl)
        out["index"], out["fine_class"], out["group"] = unknowns.index, unknowns.fine_class, unknowns.group
        save_npz(vcache / "unknown.npz", out)
    Vv["unknown"] = dict(np.load(vcache / "unknown.npz"))

    # ---- 4. comparison on the same images -------------------------------------------------------------------------
    U, D = {}, {}
    for run in ("vanilla", "proser"):
        sha_run = main_rec["checkpoints"][run]["sha256"]
        D[run] = {s: load_cached(cfg["cache_dir"], run, s, expect_sha=sha_run) for s in ("val", "test", "unknown")}
        U[run] = {s: scores.compute(D[run][s], None, T) for s in D[run]}
    D[VARIANT] = Vv
    U[VARIANT] = {s: scores.compute(Vv[s], None, T) for s in Vv}
    assert np.array_equal(D[VARIANT]["unknown"]["index"], D["vanilla"]["unknown"]["index"])
    assert np.array_equal(D[VARIANT]["test"]["labels"], D["vanilla"]["test"]["labels"])
    combined = {"thresholds": {**{r: main_rec["thresholds"][r] for r in ("vanilla", "proser")}, VARIANT: thresholds},
                "val_acceptance": {**{r: main_rec["val_acceptance"][r] for r in ("vanilla", "proser")},
                                   VARIANT: val_accept}}
    group = D["vanilla"]["unknown"]["group"]
    boot_rows = {f"{r}/{s}": {"test": U[r]["test"][s], "near": U[r]["unknown"][s][group == "near"],
                              "far": U[r]["unknown"][s][group == "far"]} for r, s in ROWS}
    print(f"bootstrap: {B} stratified resamples (seed 6304) ...", flush=True)
    boot = bootstrap_aurocs(boot_rows, B=B, seed=6304)
    t = table(ROWS, U, D, combined, boot)
    t.insert(2, "status", ["pre-registered" if r != VARIANT else "POST-HOC" for r, _ in ROWS])
    comps = auroc_comparisons(PAIRS, boot, U, D, "posthoc")
    rj = rejection_mcnemar(PAIRS, U, D, combined, "posthoc")
    cm = csa_mcnemar(CSA_PAIRS, D)
    t.to_csv(pdir / "table_posthoc_proser_bncal.csv", index=False, float_format="%.4f")
    comps.to_csv(pdir / "auroc_comparisons_posthoc.csv", index=False, float_format="%.4f")
    rj.to_csv(pdir / "rejection_mcnemar_posthoc.csv", index=False, float_format="%.6g")
    cm.to_csv(pdir / "csa_mcnemar_posthoc.csv", index=False, float_format="%.6g")
    plot_histories(Path(resolve(cfg["results_dir"])), Path(resolve(cfg.get("posthoc_runs_dir",
                   "task4/results/posthoc/runs"))), pdir / "fig_proser_bn")
    save_json({"label": rec["label"], "posthoc_frozen_at": rec["frozen_at"], "bootstrap_B": B,
               "table": t.to_dict("records"), "auroc_comparisons": comps.to_dict("records"),
               "rejection_mcnemar": rj.to_dict("records"), "csa_mcnemar": cm.to_dict("records")},
              pdir / "summary_posthoc.json")

    pd.set_option("display.width", 250)
    short = ["model", "score", "status", "CSA", "AUROC_near", "AUROC_far", "AUROC_all", "test_accept", "reject_near",
             "reject_far", "FPR95_near", "FPR95_far"]
    print("\nPOST-HOC TABLE - PROSER with BatchNorm statistics re-estimated from real training images (numbers in %)")
    print(t[short].round(2).to_string(index=False))
    print(t[["model", "score", "AUROC_near_CI", "AUROC_far_CI", "AUROC_all_CI"]].to_string(index=False))
    print("\nPaired AUROC differences (A - B), Holm within this post-hoc family x group")
    print(comps[["group", "A", "B", "AUROC_A_minus_B", "CI_low", "CI_high", "p_holm", "significant_0.05"]]
          .round(3).to_string(index=False))
    print("\nClosed-set accuracy, paired McNemar")
    print(cm.round(4).to_string(index=False))
    print(f"\nwritten to {pdir}")


def plot_histories(runs_dir, posthoc_runs_dir, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for path, label, style in ((runs_dir / "proser" / "history.json", "PROSER (as trained, pre-registered)", "-"),
                               (posthoc_runs_dir / VARIANT / "history.json",
                                "PROSER, BN statistics re-estimated (post-hoc)", "--")):
        if not path.exists():
            continue
        h = pd.DataFrame(load_json(path))
        ax[0].plot(h.epoch, h.val_acc, style, label=label)
        ax[1].plot(h.epoch, h.val_dummy_wins, style, label=label)
    ax[0].set_title("CIFAR-10 validation accuracy (10 known logits)"); ax[0].set_ylabel("%")
    ax[1].set_title("% of validation images where the dummy beats every known class"); ax[1].set_ylabel("%")
    for a in ax:
        a.set_xlabel("fine-tuning epoch"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task4/configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--refreeze-reason", default=None)
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set), a.bootstrap, a.refreeze_reason)
