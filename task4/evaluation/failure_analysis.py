"""Failure analysis: unknowns that the Vanilla model ACCEPTS as known under the Vanilla-MLS threshold.

  python -m task4.evaluation.failure_analysis --view main                   # the required failure analysis
  python -m task4.evaluation.failure_analysis --view confident_borderline   # supplementary view C
  python -m task4.evaluation.failure_analysis --view random                 # supplementary view B
  (plus --config task4/configs/base.yaml --set data_root=... cache_dir=... as for the other scripts)

Handout: "Using the vanilla MLS threshold, inspect at least three incorrectly accepted near unknowns and three
incorrectly accepted far unknowns. Record the unknown class, predicted CIFAR-10 class, score, and threshold, and
distinguish semantically plausible confusions from surprising failures." These observations are for analysis only
and must not be used to revise the model, score, hyperparameters or threshold. That is guaranteed here: everything
is read from the freeze record, and this script writes nothing that any earlier stage reads.

Every selection rule was fixed before any result existed (evaluation/preregistered.py, copied into the freeze
record) and is read FROM THE RECORD:
  main                  the most confidently accepted image of each class (lowest u_MLS), ordered by u_MLS, first 6
                        per group; topped up to 3 if fewer than 3 classes have an accepted image
  confident_borderline  (view C) per class: the most confidently accepted image AND the borderline one (the
                        accepted image closest to tau)
  random                (view B) 6 accepted images per group drawn with a seeded random generator: typical cases

This script does NOT label anything plausible or surprising - that judgement is yours. Every view adds its images
to ONE sheet, failures/your_judgement.csv (columns your_call / your_reason, plus in_views saying which views show
the image). Rows are only ever added and in_views updated; your entries are never touched. A SUGGESTED map is
applied separately, into its own folder, by evaluation/apply_plausibility_map.py.

Outputs (results/final/failures/):
  main:                  accepted_unknowns_vanilla_mls.csv, selected_failures.csv, acceptance_by_class_vanilla_mls.csv,
                         failures_vanilla_mls.(png|pdf), summary.json
  confident_borderline:  view_confident_borderline/ selected.csv, grid.(png|pdf), summary.json
  random:                view_random_sample/       selected.csv, grid.(png|pdf), summary.json
  every view:            your_judgement.csv (shared)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from common.io import apply_overrides, load_config, resolve, save_json
from task4 import scores
from task4.audit import verify_frozen
from task4.data.cifar10 import CIFAR10_CLASSES, build_transform
from task4.data.cifar100_unknowns import FAR, NEAR, load_unknowns
from task4.evaluation.plots import failure_grid
from task4.extract_outputs import FROZEN_FILE, load_cached
from task4.scores import mahalanobis as maha

VIEWS = ["main", "confident_borderline", "random"]
COLS = ["group", "fine_class", "index", "pred", "second_choice", "u", "tau", "margin_below_tau", "max_logit",
        "max_softmax"]
SHEET_COLS = ["group", "fine_class", "index", "pred", "second_choice", "u", "tau"]


def accepted_unknowns(cfg, rec, model, score):
    """Every unknown's score under the frozen model/score, and the accepted subset (u <= tau), most confident first."""
    tau = rec["thresholds"][model][score]
    df = pd.read_csv(Path(resolve(cfg.get("scores_dir", "task4/results/scores"))) / f"{model}_unknown.csv")
    d = load_cached(cfg["cache_dir"], model, "unknown", expect_sha=rec["checkpoints"][model]["sha256"])
    assert np.array_equal(df["index"].to_numpy(), d["index"])
    z = d["logits"].astype(np.float64)
    df["pos"] = np.arange(len(df))
    df["max_logit"] = z.max(1)
    df["max_softmax"] = softmax(z, axis=1).max(1)
    df["second_choice"] = [CIFAR10_CLASSES[i] for i in np.argsort(-z, axis=1)[:, 1]]
    # recompute u from the cached logits at full precision (the CSV holds 8 significant digits)
    u_all = scores.compute(d, maha.load(rec["mahalanobis_stats"][model]["path"]),
                           rec["protocol"]["proser_detection_temperature"])
    df["u"] = u_all[score]
    assert np.allclose(df["u"], df[f"u_{score}"], rtol=1e-6, atol=1e-9), "scores CSV disagrees with the cache"
    df["tau"] = tau
    df["margin_below_tau"] = tau - df["u"]                  # > 0 = accepted; larger = more confidently accepted
    acc = df[df["u"] <= tau].sort_values(["u", "index"])
    return df, acc, d, tau


def select_main(acc, n_per):
    sel = []
    for g in ("near", "far"):
        a = acc[acc["group"] == g]
        pick = a.loc[a.groupby("fine_class")["u"].idxmin()].sort_values("u").head(n_per)   # one per class
        if len(pick) < 3:                                  # top up to 3 with the next most confident images
            pick = pd.concat([pick, a.drop(pick.index).head(3 - len(pick))])
        sel.append(pick)
    return pd.concat(sel)


def select_confident_borderline(acc):
    rows = []
    for g, names in (("near", NEAR), ("far", FAR)):
        for c in names:
            a = acc[acc["fine_class"] == c]
            if len(a) == 0:
                continue
            most, border = a.loc[a["u"].idxmin()], a.loc[a["u"].idxmax()]
            rows.append({**most.to_dict(), "role": "most confident"})
            if border["index"] != most["index"]:
                rows.append({**border.to_dict(), "role": "borderline"})
    return pd.DataFrame(rows)


def select_random(acc, n, seed):
    rng = np.random.default_rng(seed)
    sel = []
    for g in ("near", "far"):
        a = acc[acc["group"] == g].sort_values(["u", "index"]).reset_index(drop=True)
        take = rng.choice(len(a), size=min(n, len(a)), replace=False) if len(a) else np.array([], int)
        sel.append(a.iloc[np.sort(take)])                  # sorted positions = ascending u
    return pd.concat(sel)


def update_sheet(path, rows, view):
    """Add this view's images to the shared judgement sheet. Existing rows keep your_call / your_reason."""
    new = rows[SHEET_COLS].drop_duplicates("index")
    if path.exists():
        j = pd.read_csv(path, dtype={"your_call": str, "your_reason": str, "in_views": str}, keep_default_na=False)
    else:
        j = pd.DataFrame(columns=SHEET_COLS + ["in_views", "your_call", "your_reason"])
    known = set(j["index"].astype(int)) if len(j) else set()
    added = 0
    for _, r in new.iterrows():
        if int(r["index"]) in known:
            m = j["index"].astype(int) == int(r["index"])
            views = [v for v in str(j.loc[m, "in_views"].iloc[0]).split(";") if v]
            if view not in views:
                j.loc[m, "in_views"] = ";".join(views + [view])
        else:
            j = pd.concat([j, pd.DataFrame([{**r.to_dict(), "in_views": view, "your_call": "", "your_reason": ""}])],
                          ignore_index=True)
            added += 1
    j.to_csv(path, index=False, float_format="%.6g")
    filled = int((j["your_call"].astype(str).str.strip() != "").sum())
    print(f"{path.name}: {added} new row(s) from view '{view}'; {len(j)} rows in total, {filled} already judged "
          "(your entries are never overwritten)")


def main(cfg, view):
    ffile = cfg.get("frozen_file", FROZEN_FILE)
    rec = verify_frozen(ffile, quiet=True)          # load_unknowns below verifies (and reports) again
    fa = rec["preregistered"]["failure_analysis"]
    views = rec["preregistered"]["failure_views"]
    model, score = fa["model"], fa["score"]
    base = Path(resolve(cfg.get("final_dir", "task4/results/final"))) / "failures"
    base.mkdir(parents=True, exist_ok=True)
    df, acc, d, tau = accepted_unknowns(cfg, rec, model, score)

    if view == "main":
        out, rule = base, fa["rule"]
        sel = select_main(acc, fa["n_per_group"])
        acc[COLS].to_csv(out / "accepted_unknowns_vanilla_mls.csv", index=False, float_format="%.6g")
        sel[COLS].to_csv(out / "selected_failures.csv", index=False, float_format="%.6g")
        pc = pd.DataFrame([{"group": "near" if c in NEAR else "far", "class": c,
                            "accepted_of_100": int((acc["fine_class"] == c).sum()),
                            "absorbed_by": ", ".join(f"{k} {v}" for k, v in
                                                     acc[acc["fine_class"] == c]["pred"].value_counts().head(3).items())}
                           for c in NEAR + FAR])
        pc.to_csv(out / "acceptance_by_class_vanilla_mls.csv", index=False)
    elif view == "confident_borderline":
        out, rule = base / "view_confident_borderline", views["confident_borderline"]["rule"]
        sel = select_confident_borderline(acc)
        out.mkdir(exist_ok=True)
        sel[["role"] + COLS].to_csv(out / "selected.csv", index=False, float_format="%.6g")
    elif view == "random":
        vr = views["random"]
        out, rule = base / "view_random_sample", vr["rule"]
        sel = select_random(acc, vr["n_per_group"], vr["seed"])
        out.mkdir(exist_ok=True)
        sel[COLS].to_csv(out / "selected.csv", index=False, float_format="%.6g")
    else:
        raise SystemExit(f"unknown view {view!r}; choose from {VIEWS}")

    update_sheet(base / "your_judgement.csv", sel, view)

    # ---- image grid (the only step that reads CIFAR-100 pixels: verified + logged access) ----------------------
    unknowns = load_unknowns(cfg["data_root"], build_transform("eval"), ffile)
    assert np.array_equal(unknowns.index, d["index"])
    if view == "confident_borderline":
        imgs, titles, labels = [], [], []
        for g, names in (("near", NEAR), ("far", FAR)):
            for role in ("most confident", "borderline"):
                row_i, row_t = [], []
                for c in names:
                    r = sel[(sel["fine_class"] == c) & (sel["role"] == role)] if len(sel) else sel
                    if len(r) == 0:
                        only = len(sel) and ((sel["fine_class"] == c).any())
                        row_i.append(None)
                        row_t.append(f"{c}\n(same image as above)" if only else f"{c}\n(none accepted)")
                    else:
                        r = r.iloc[0]
                        row_i.append(unknowns.raw_image(int(r["pos"])))
                        row_t.append(f"{c} -> {r['pred']}\nu={r['u']:.2f}")
                imgs.append(row_i); titles.append(row_t); labels.append(f"{g}\n{role}")
        failure_grid(imgs, titles, out / "grid", f"View C - most confident and borderline accepted image per class "
                     f"(Vanilla-MLS, accept if u <= tau = {tau:.3f})", row_labels=labels)
    else:
        imgs, titles = [], []
        for g in ("near", "far"):
            s = sel[sel["group"] == g]
            imgs.append([unknowns.raw_image(int(p)) for p in s["pos"]])
            titles.append([f"{r.fine_class} -> {r.pred}\nu={r.u:.2f} (tau={tau:.2f})" for r in s.itertuples()])
        stem, what = ((out / "failures_vanilla_mls", "most confidently accepted image per class") if view == "main"
                      else (out / "grid", "View B - random sample of accepted images"))
        failure_grid(imgs, titles, stem, f"{what} (Vanilla-MLS, accept if u <= tau = {tau:.3f}); top row near, "
                     "bottom row far", row_labels=["near", "far"])

    summary = {"view": view, "model": model, "score": score, "tau": tau, "rule": rule,
               "accepted_near": int((acc["group"] == "near").sum()), "accepted_far": int((acc["group"] == "far").sum()),
               "selected": sel[(["role"] if "role" in sel else []) + COLS].to_dict("records")}
    save_json(summary, out / "summary.json")

    print(f"\nview: {view}\nrule: {rule}")
    print(f"Vanilla-MLS threshold tau = {tau:.4f}  (accept if u = -max logit <= tau, i.e. max logit >= {-tau:.4f})")
    print(f"accepted unknowns: near {summary['accepted_near']}/800, far {summary['accepted_far']}/800")
    if view == "main":
        print("\nper class (100 images each, so the count is also a percentage):")
        print(pc.to_string(index=False))
    print("\nselected images:")
    print(sel[(["role"] if "role" in sel else []) + COLS].round(3).to_string(index=False))
    if view == "main":
        for g in ("near", "far"):
            k = int((sel["group"] == g).sum())
            if k < 3:
                print(f"\nNOTE: only {k} {g} unknown(s) were accepted at all - fewer than the 3 the handout asks for, "
                      f"because Vanilla-MLS rejected every other {g} unknown. Report that fact.")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task4/configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--view", choices=VIEWS, default="main")
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set), a.view)
