"""Final open-set evaluation: the two required tables, their uncertainty, score agreement, per-class behaviour
and the score figure.

  python -m task4.evaluate_osr --config task4/configs/base.yaml [--set ...]

Runs only after `task4.freeze` and `extract_outputs --stage unknown`. It reads the frozen thresholds and
Mahalanobis statistics from the freeze record (verified first) and recomputes nothing that could be tuned: the
validation thresholds are recomputed only to CHECK they equal the frozen ones.

Outputs in results/final/:
  table1_posthoc_scores_vanilla.csv    MSP / MLS / Energy / Mahalanobis on the frozen Vanilla model
  table2_models_mls.csv                Vanilla / GCSC / PROSER with MLS + PROSER placeholder-detection row
  all_models_all_scores.csv            every model x every score (supplementary)
  auroc_comparisons.csv                paired bootstrap AUROC differences (Holm within table x group)
  rejection_mcnemar.csv                paired McNemar tests on unknown rejection at the frozen thresholds
  csa_mcnemar.csv                      paired McNemar tests on closed-set accuracy
  score_agreement_spearman.csv         rank correlation of the four Vanilla scores within known / near / far
  decision_overlap_unknowns.csv        which unknowns each pair of Vanilla scores rejects (both / only one / neither)
  acceptance_by_class.csv              per CIFAR-100 class: % accepted + the CIFAR-10 class that absorbs most
  absorption_counts.csv                accepted unknowns: CIFAR-100 class x predicted CIFAR-10 class counts
  summary.json                         everything above in one file, with the freeze timestamp
  figures/fig_vanilla_scores.(png|pdf) score distributions + ROC curves for MSP, MLS, Mahalanobis
  figures/fig_acceptance_by_class.(png|pdf)
results/scores/<run>_<split>.csv       per-image scores (val, test, unknown) - every number above traces to these
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.io import apply_overrides, load_config, resolve, save_json
from task4 import scores
from task4.audit import verify_frozen
from task4.data.cifar10 import CIFAR10_CLASSES
from task4.data.cifar100_unknowns import FAR, NEAR
from task4.evaluation import plots
from task4.evaluation.metrics import auroc, csa, osr_metrics
from task4.evaluation.stats import boot_p, bootstrap_aurocs, ci, holm, mcnemar_exact, wilson
from task4.evaluation.thresholds import fit_threshold
from task4.extract_outputs import FROZEN_FILE, RUNS, load_cached
from task4.scores import mahalanobis as maha

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
NICE = {"msp": "MSP", "mls": "MLS", "energy": "Energy", "mahalanobis": "Mahalanobis", "proser": "PROSER-detect"}


def load_all(cfg, rec):
    """U[run][split][score] = unknownness; D[run][split] = cached arrays."""
    T = rec["protocol"]["proser_detection_temperature"]
    U, D = {}, {}
    for run in RUNS:
        sha = rec["checkpoints"][run]["sha256"]
        stats = maha.load(rec["mahalanobis_stats"][run]["path"])
        U[run], D[run] = {}, {}
        for split in ["val", "test", "unknown"]:
            D[run][split] = load_cached(cfg["cache_dir"], run, split, expect_sha=sha)
            U[run][split] = scores.compute(D[run][split], stats, T)
        for s, u in U[run]["val"].items():                         # thresholds must equal the frozen ones
            tau = rec["thresholds"][run][s]
            assert abs(fit_threshold(u) - tau) <= 1e-9 * max(1.0, abs(tau)), f"{run}/{s}: threshold mismatch"
    for run in RUNS[1:]:                                           # every model saw the same images
        assert np.array_equal(D[run]["unknown"]["index"], D["vanilla"]["unknown"]["index"])
        assert np.array_equal(D[run]["test"]["labels"], D["vanilla"]["test"]["labels"])
    return U, D


def save_score_csvs(U, D, out):
    out.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        for split in ["val", "test", "unknown"]:
            d = D[run][split]
            df = pd.DataFrame({"index": d["index"]})
            if split == "unknown":
                df["group"], df["fine_class"] = d["group"], d["fine_class"]
            else:
                df["label"] = [CIFAR10_CLASSES[i] for i in d["labels"]]
            df["pred"] = [CIFAR10_CLASSES[i] for i in d["logits"].argmax(1)]
            for s, u in U[run][split].items():
                df[f"u_{s}"] = u
            df.to_csv(out / f"{run}_{split}.csv", index=False, float_format="%.8g")


def table(rows, U, D, rec, boot):
    group = D["vanilla"]["unknown"]["group"]
    n_near, n_far, n_test = int((group == "near").sum()), int((group == "far").sum()), len(D["vanilla"]["test"]["labels"])
    out = []
    for run, s in rows:
        tau = rec["thresholds"][run][s]
        m = osr_metrics(U[run]["test"][s], U[run]["unknown"][s], group, tau)
        acc = csa(D[run]["test"]["logits"], D[run]["test"]["labels"])
        r = {"model": run, "score": NICE[s], "CSA": acc, "tau": tau, "val_accept": rec["val_acceptance"][run][s]}
        for g in ("near", "far", "all"):
            r[f"AUROC_{g}"] = m[f"auroc_{g}"]
            if boot is not None:
                lo, hi = ci(boot[f"{run}/{s}"][g])
                r[f"AUROC_{g}_CI"] = f"[{lo:.2f}, {hi:.2f}]"
        r["test_accept"] = m["test_accept"]
        k = round(m["test_accept"] * n_test / 100)
        r["test_accept_CI"] = "[{:.2f}, {:.2f}]".format(*wilson(k, n_test))
        for g, n in (("near", n_near), ("far", n_far), ("all", n_near + n_far)):
            r[f"reject_{g}"] = m[f"reject_{g}"]
            r[f"reject_{g}_CI"] = "[{:.1f}, {:.1f}]".format(*wilson(round(m[f"reject_{g}"] * n / 100), n))
        for g in ("near", "far", "all"):
            r[f"FPR95_{g}"] = m[f"fpr95_{g}"]
        out.append(r)
    return pd.DataFrame(out)


def auroc_comparisons(pairs, boot, U, D, label):
    group = D["vanilla"]["unknown"]["group"]
    rows = []
    for g in ("near", "far", "all"):
        fam = []
        for a, b in pairs:
            ka, kb = "/".join(a), "/".join(b)
            diff = boot[ka][g] - boot[kb][g]
            ta, tb = U[a[0]]["test"][a[1]], U[b[0]]["test"][b[1]]
            ua, ub = U[a[0]]["unknown"][a[1]], U[b[0]]["unknown"][b[1]]
            sel = slice(None) if g == "all" else (group == g)
            point = 100 * (auroc(ta, ua[sel]) - auroc(tb, ub[sel]))
            lo, hi = ci(diff)
            fam.append({"table": label, "group": g, "A": f"{a[0]}-{NICE[a[1]]}", "B": f"{b[0]}-{NICE[b[1]]}",
                        "AUROC_A_minus_B": point, "CI_low": lo, "CI_high": hi, "p_boot": boot_p(diff)})
        adj = holm([r["p_boot"] for r in fam])
        for r, p in zip(fam, adj):
            r["p_holm"] = p
            r["significant_0.05"] = bool(p < 0.05)
        rows += fam
    return pd.DataFrame(rows)


def rejection_mcnemar(pairs, U, D, rec, label):
    group = D["vanilla"]["unknown"]["group"]
    rows = []
    for g in ("near", "far"):
        fam = []
        for a, b in pairs:
            ra = U[a[0]]["unknown"][a[1]][group == g] > rec["thresholds"][a[0]][a[1]]
            rb = U[b[0]]["unknown"][b[1]][group == g] > rec["thresholds"][b[0]][b[1]]
            n01, n10, p = mcnemar_exact(ra, rb)
            fam.append({"table": label, "group": g, "A": f"{a[0]}-{NICE[a[1]]}", "B": f"{b[0]}-{NICE[b[1]]}",
                        "reject_A": 100 * ra.mean(), "reject_B": 100 * rb.mean(),
                        "only_A_rejects": n10, "only_B_rejects": n01, "p_exact": p})
        for r, p in zip(fam, holm([r["p_exact"] for r in fam])):
            r["p_holm"] = p
            r["significant_0.05"] = bool(p < 0.05)
        rows += fam
    return pd.DataFrame(rows)


def csa_mcnemar(pairs, D):
    y = D["vanilla"]["test"]["labels"]
    rows = []
    for a, b in pairs:
        oa, ob = D[a]["test"]["logits"].argmax(1) == y, D[b]["test"]["logits"].argmax(1) == y
        n01, n10, p = mcnemar_exact(oa, ob)
        rows.append({"A": a, "B": b, "CSA_A": 100 * oa.mean(), "CSA_B": 100 * ob.mean(),
                     "diff": 100 * (oa.mean() - ob.mean()), "only_A_correct": n10, "only_B_correct": n01,
                     "p_exact": p})
    for r, p in zip(rows, holm([r["p_exact"] for r in rows])):
        r["p_holm"] = p
        r["significant_0.05"] = bool(p < 0.05)
    return pd.DataFrame(rows)


def agreement(U, D, rec):
    """How the four Vanilla scores agree: rank correlation, and overlap of their reject decisions."""
    group = D["vanilla"]["unknown"]["group"]
    posthoc = ["msp", "mls", "energy", "mahalanobis"]
    sets = {"known_test": {s: U["vanilla"]["test"][s] for s in posthoc},
            "near": {s: U["vanilla"]["unknown"][s][group == "near"] for s in posthoc},
            "far": {s: U["vanilla"]["unknown"][s][group == "far"] for s in posthoc}}
    corr = []
    for name, d in sets.items():
        for i, a in enumerate(posthoc):
            for b in posthoc[i + 1:]:
                corr.append({"set": name, "A": NICE[a], "B": NICE[b], "spearman": spearmanr(d[a], d[b])[0]})
    overlap = []
    for g in ("near", "far"):
        rej = {s: U["vanilla"]["unknown"][s][group == g] > rec["thresholds"]["vanilla"][s] for s in posthoc}
        for i, a in enumerate(posthoc):
            for b in posthoc[i + 1:]:
                overlap.append({"group": g, "A": NICE[a], "B": NICE[b],
                                "both_reject_%": 100 * (rej[a] & rej[b]).mean(),
                                "only_A_rejects_%": 100 * (rej[a] & ~rej[b]).mean(),
                                "only_B_rejects_%": 100 * (~rej[a] & rej[b]).mean(),
                                "both_accept_%": 100 * (~rej[a] & ~rej[b]).mean()})
    return pd.DataFrame(corr), pd.DataFrame(overlap)


def per_class(rows, U, D, rec):
    d = D["vanilla"]["unknown"]
    fine, group = d["fine_class"], d["group"]
    acc_rows, absorb = [], []
    for run, s in rows:
        acc_mask = U[run]["unknown"][s] <= rec["thresholds"][run][s]
        pred = D[run]["unknown"]["logits"].argmax(1)
        for c in NEAR + FAR:
            m = fine == c
            a = acc_mask & m
            counts = np.bincount(pred[a], minlength=10)
            top = int(counts.argmax()) if a.any() else -1
            acc_rows.append({"model": run, "score": NICE[s], "group": "near" if c in NEAR else "far",
                             "class": c, "accepted_%": 100 * a.sum() / m.sum(),
                             "top_absorber": CIFAR10_CLASSES[top] if top >= 0 else "",
                             "top_absorber_share_%": 100 * counts[top] / a.sum() if a.any() else np.nan,
                             "argmax_all_images": CIFAR10_CLASSES[int(np.bincount(pred[m], minlength=10).argmax())]})
            absorb.append({"model": run, "score": NICE[s], "class": c,
                           **{k: int(v) for k, v in zip(CIFAR10_CLASSES, counts)}})
    return pd.DataFrame(acc_rows), pd.DataFrame(absorb)


def main(cfg, B):
    rec = verify_frozen(cfg.get("frozen_file", FROZEN_FILE))
    pre = rec["preregistered"]
    t1_rows = [tuple(r) for r in pre["table1_rows"]]
    t2_rows = [tuple(r) for r in pre["table2_rows"]]
    final = Path(resolve(cfg.get("final_dir", "task4/results/final")))
    (final / "figures").mkdir(parents=True, exist_ok=True)

    U, D = load_all(cfg, rec)
    save_score_csvs(U, D, Path(resolve(cfg.get("scores_dir", "task4/results/scores"))))
    group = D["vanilla"]["unknown"]["group"]
    print(f"frozen at {rec['frozen_at']}; test {len(D['vanilla']['test']['labels'])} knowns, unknowns: "
          f"{int((group == 'near').sum())} near + {int((group == 'far').sum())} far\n")

    # ---- bootstrap on the union of table rows (same resamples for every row -> paired) -------------------
    boot_rows = {f"{r}/{s}": {"test": U[r]["test"][s], "near": U[r]["unknown"][s][group == "near"],
                              "far": U[r]["unknown"][s][group == "far"]} for r, s in dict.fromkeys(t1_rows + t2_rows)}
    print(f"bootstrap: {B} stratified resamples of the test knowns and each unknown group (seed 6304) ...", flush=True)
    boot = bootstrap_aurocs(boot_rows, B=B, seed=6304) if B > 0 else None

    t1 = table(t1_rows, U, D, rec, boot)
    t2 = table(t2_rows, U, D, rec, boot)
    full = table([(r, s) for r in RUNS for s in U[r]["test"]], U, D, rec, None)
    t1.to_csv(final / "table1_posthoc_scores_vanilla.csv", index=False, float_format="%.4f")
    t2.to_csv(final / "table2_models_mls.csv", index=False, float_format="%.4f")
    full.to_csv(final / "all_models_all_scores.csv", index=False, float_format="%.4f")

    comps = pd.DataFrame()
    if boot is not None:
        c1 = auroc_comparisons([tuple(map(tuple, p)) for p in pre["comparisons_table1"]], boot, U, D, "table1")
        c2 = auroc_comparisons([tuple(map(tuple, p)) for p in pre["comparisons_table2"]], boot, U, D, "table2")
        comps = pd.concat([c1, c2], ignore_index=True)
        comps.to_csv(final / "auroc_comparisons.csv", index=False, float_format="%.4f")
    rj = pd.concat([rejection_mcnemar([tuple(map(tuple, p)) for p in pre["comparisons_table1"]], U, D, rec, "table1"),
                    rejection_mcnemar([tuple(map(tuple, p)) for p in pre["comparisons_table2"]], U, D, rec, "table2")],
                   ignore_index=True)
    rj.to_csv(final / "rejection_mcnemar.csv", index=False, float_format="%.6g")
    cm = csa_mcnemar([tuple(p) for p in pre["csa_comparisons"]], D)
    cm.to_csv(final / "csa_mcnemar.csv", index=False, float_format="%.6g")
    corr, overlap = agreement(U, D, rec)
    corr.to_csv(final / "score_agreement_spearman.csv", index=False, float_format="%.4f")
    overlap.to_csv(final / "decision_overlap_unknowns.csv", index=False, float_format="%.2f")
    pc, ab = per_class(list(dict.fromkeys(t1_rows + t2_rows)), U, D, rec)
    pc.to_csv(final / "acceptance_by_class.csv", index=False, float_format="%.2f")
    ab.to_csv(final / "absorption_counts.csv", index=False)

    plots.score_figure(U["vanilla"], group, rec["thresholds"]["vanilla"], final / "figures" / "fig_vanilla_scores")
    plots.acceptance_by_class_figure(pc, t2_rows, NICE, final / "figures" / "fig_acceptance_by_class")

    save_json({"frozen_at": rec["frozen_at"], "bootstrap_B": B,
               "table1": t1.to_dict("records"), "table2": t2.to_dict("records"),
               "auroc_comparisons": comps.to_dict("records"), "rejection_mcnemar": rj.to_dict("records"),
               "csa_mcnemar": cm.to_dict("records")}, final / "summary.json")

    # ---- console report ----------------------------------------------------------------------------------------
    short = ["model", "score", "CSA", "AUROC_near", "AUROC_far", "AUROC_all", "test_accept", "reject_near",
             "reject_far", "FPR95_near", "FPR95_far"]
    print("\nTABLE 1 - post-hoc scores on the frozen Vanilla model (all numbers in %)")
    print(t1[short].round(2).to_string(index=False))
    print("\nTABLE 2 - Vanilla vs GCSC vs PROSER (MLS) + PROSER placeholder detection (all numbers in %)")
    print(t2[short].round(2).to_string(index=False))
    if boot is not None:
        print("\nAUROC 95% bootstrap intervals")
        print(pd.concat([t1, t2])[["model", "score", "AUROC_near_CI", "AUROC_far_CI", "AUROC_all_CI"]]
              .drop_duplicates().to_string(index=False))
        print("\nPaired AUROC differences (A - B, points of AUROC), Holm within table x group")
        print(comps[["table", "group", "A", "B", "AUROC_A_minus_B", "CI_low", "CI_high", "p_holm",
                     "significant_0.05"]].round(3).to_string(index=False))
    print("\nClosed-set accuracy, paired McNemar (Holm over 3 comparisons)")
    print(cm.round(4).to_string(index=False))
    print(f"\nwritten to {final}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task4/configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set), a.bootstrap)
