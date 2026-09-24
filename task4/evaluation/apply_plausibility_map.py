"""Apply the SUGGESTED plausibility map to the failure analysis - kept separate on purpose.

  python -m task4.evaluation.apply_plausibility_map
  python -m task4.evaluation.apply_plausibility_map --map task4/configs/plausibility_map.yaml

The handout asks to "distinguish semantically plausible confusions from surprising failures". That is a judgement
about images, so it is NOT in the freeze record and NOT applied by failure_analysis.py. Instead:

  1. failure_analysis.py selects the failures (pre-registered rules: the main view, plus views C and B) and adds
     them to one blank sheet, your_judgement.csv.
  2. You look at the images and fill in your_call (plausible / surprising) and your_reason.
  3. This script applies the suggested map (configs/plausibility_map.yaml) and, if your_judgement.csv has
     entries, lists where your calls and the map agree and disagree.

It reads only the CSVs that failure_analysis.py wrote - no model, no GPU, no CIFAR-100 access - so you can edit
the map and re-run it in seconds, as often as you like. It changes no metric, table, threshold or model.

Outputs in results/final/failures/suggested_map/ (a separate folder, so nothing is mixed with the neutral results):
  map_used.yaml                     the exact map applied (copy), for the record
  selected_failures_labelled.csv    the main selection + the map's label
  view_confident_borderline_labelled.csv, view_random_sample_labelled.csv   the same for views C and B, if run
  accepted_unknowns_labelled.csv    every accepted unknown + the map's label
  plausibility_by_class.csv         per unknown class: accepted, plausible, surprising, top absorbing classes
  judgement_vs_map.csv              only if you filled your_judgement.csv: your call vs the map, row by row
  summary.json
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
import yaml

from common.io import apply_overrides, load_config, resolve, save_json

# Copied from task4/data/cifar10.py and cifar100_unknowns.py so this script runs without PyTorch installed.
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
NEAR = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]
DEFAULT_MAP = "task4/configs/plausibility_map.yaml"


def load_map(path):
    m = yaml.safe_load(open(resolve(path)))
    flat = {**(m.get("near") or {}), **(m.get("far") or {})}
    missing = [c for c in NEAR + FAR if c not in flat]
    extra = [c for c in flat if c not in NEAR + FAR]
    bad = {c: [t for t in (v or []) if t not in CIFAR10_CLASSES] for c, v in flat.items()}
    bad = {c: v for c, v in bad.items() if v}
    wrong_group = [c for c in (m.get("near") or {}) if c not in NEAR] + [c for c in (m.get("far") or {}) if c not in FAR]
    problems = []
    if missing: problems.append(f"classes missing from the map: {missing}")
    if extra: problems.append(f"unknown class names in the map: {extra}")
    if bad: problems.append(f"not CIFAR-10 class names: {bad}")
    if wrong_group: problems.append(f"classes under the wrong group (near/far is fixed by the handout): {wrong_group}")
    if problems:
        raise SystemExit("plausibility map problems:\n  " + "\n  ".join(problems))
    return {c: list(v or []) for c, v in flat.items()}, m.get("author", "")


def label(df, pmap):
    df = df.copy()
    df["plausible_targets"] = ["/".join(pmap[c]) or "-" for c in df["fine_class"]]
    df["map_label"] = ["plausible" if p in pmap[c] else "surprising" for c, p in zip(df["fine_class"], df["pred"])]
    return df


def normalise_call(x):
    x = str(x).strip().lower() if pd.notna(x) else ""
    if not x:
        return None
    if x.startswith("p"):
        return "plausible"
    if x.startswith("s"):
        return "surprising"
    return "INVALID: " + x


def main(cfg, map_path):
    fdir = Path(resolve(cfg.get("final_dir", "task4/results/final"))) / "failures"
    out = fdir / "suggested_map"
    out.mkdir(parents=True, exist_ok=True)
    pmap, author = load_map(map_path)
    map_sha = hashlib.sha256(open(resolve(map_path), "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    shutil.copyfile(resolve(map_path), out / "map_used.yaml")

    sel = label(pd.read_csv(fdir / "selected_failures.csv"), pmap)
    acc = label(pd.read_csv(fdir / "accepted_unknowns_vanilla_mls.csv"), pmap)
    sel.to_csv(out / "selected_failures_labelled.csv", index=False, float_format="%.6g")
    acc.to_csv(out / "accepted_unknowns_labelled.csv", index=False, float_format="%.6g")
    view_tables = {}
    for v in ("view_confident_borderline", "view_random_sample"):
        if (fdir / v / "selected.csv").exists():
            view_tables[v] = label(pd.read_csv(fdir / v / "selected.csv"), pmap)
            view_tables[v].to_csv(out / f"{v}_labelled.csv", index=False, float_format="%.6g")

    rows = []
    for c in NEAR + FAR:
        a = acc[acc["fine_class"] == c]
        rows.append({"group": "near" if c in NEAR else "far", "class": c, "plausible_targets": "/".join(pmap[c]) or "-",
                     "accepted_of_100": len(a), "plausible": int((a["map_label"] == "plausible").sum()),
                     "surprising": int((a["map_label"] == "surprising").sum()),
                     "absorbed_by": ", ".join(f"{k} {v}" for k, v in a["pred"].value_counts().head(3).items())})
    pc = pd.DataFrame(rows)
    pc.to_csv(out / "plausibility_by_class.csv", index=False)

    summary = {"map_file": map_path, "map_author": author, "map_sha256": map_sha,
               "note": "Suggested map - not pre-registered, not in the freeze record, changes no metric.",
               "accepted": {g: int((acc["group"] == g).sum()) for g in ("near", "far")},
               "accepted_plausible": {g: int(((acc["group"] == g) & (acc["map_label"] == "plausible")).sum())
                                      for g in ("near", "far")},
               "selected_labels": sel[["group", "fine_class", "index", "pred", "map_label"]].to_dict("records"),
               "views_labelled": sorted(view_tables)}

    # ---- your calls vs the map (only if you have filled your_judgement.csv) -------------------------------------
    jfile = fdir / "your_judgement.csv"
    if jfile.exists():
        j = pd.read_csv(jfile, dtype={"your_call": str, "your_reason": str, "in_views": str}, keep_default_na=False)
        j["your_call_clean"] = [normalise_call(x) for x in j["your_call"]]
        filled = label(j[j["your_call_clean"].notna()], pmap)
        if len(filled):
            cmp_ = filled.copy()
            cmp_["agree"] = cmp_["your_call_clean"] == cmp_["map_label"]
            cmp_ = cmp_[["group", "fine_class", "index", "pred", "in_views", "your_call_clean", "map_label", "agree",
                         "plausible_targets", "your_reason"]].rename(columns={"your_call_clean": "your_call"})
            cmp_.to_csv(out / "judgement_vs_map.csv", index=False)
            summary["judgement"] = {"filled": int(len(filled)), "of": int(len(j)), "agree": int(cmp_["agree"].sum()),
                                    "invalid": int(cmp_["your_call"].str.startswith("INVALID").sum())}
        else:
            summary["judgement"] = {"filled": 0, "of": int(len(j))}
    save_json(summary, out / "summary.json")

    # ---- console report -------------------------------------------------------------------------------------------
    print(f"SUGGESTED plausibility map: {map_path}  ({author or 'no author line'})")
    print("  Not pre-registered and not in the freeze record; it only labels the failures and changes no metric.\n")
    print("the map:")
    for g, names in (("near", NEAR), ("far", FAR)):
        print(f"  {g}: " + "; ".join(f"{c} -> {'/'.join(pmap[c]) or '(none)'}" for c in names))
    print("\nmain selection with the map's label:")
    print(sel[["group", "fine_class", "pred", "u", "tau", "plausible_targets", "map_label"]].round(3).to_string(index=False))
    for v, t in view_tables.items():
        print(f"\n{v} with the map's label:")
        print(t[[c for c in ["role", "group", "fine_class", "pred", "u", "plausible_targets", "map_label"] if c in t]]
              .round(3).to_string(index=False))
    print("\nper class, all accepted unknowns (100 images per class, so counts are percentages):")
    print(pc.to_string(index=False))
    jd = summary.get("judgement")
    if jd is None:
        print("\nno your_judgement.csv found.")
    elif jd["filled"] == 0:
        print(f"\nyour_judgement.csv: none of the {jd['of']} rows filled yet. Fill your_call (plausible / surprising) "
              "and your_reason, then re-run this script to compare with the map.")
    else:
        print(f"\nyour calls vs the map: {jd['agree']}/{jd['filled']} agree ({jd['of']} rows in the sheet)")
        print(pd.read_csv(out / "judgement_vs_map.csv", keep_default_na=False).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task4/configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--map", default=DEFAULT_MAP)
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set), a.map)
