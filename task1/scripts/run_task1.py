"""Task 1 driver: features -> heads -> every table/figure in the 'Required Evidence' list.

Order of operations (run from the repository root):
  python -m task1.data.make_subset        --config task1/configs/task1.yaml
  python -m task1.data.make_cue_conflicts --config task1/configs/task1.yaml
  #   ... review task1/results/cue_conflicts/sheets/*.png, edit review.csv ...
  python -m task1.scripts.run_task1       --config task1/configs/task1.yaml

Design: transformation generation is separated from evaluation. Deterministic
interventions (grayscale, hue, translation) are recomputed identically for every model;
the random ones (patch permutations, cue conflicts) are generated ONCE and saved, so
all backbones see byte-identical inputs.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.io import load_config, load_json, resolve, save_json
from common.seed import get_device, set_seed
from task1.analysis.evaluate_bias import decision_type, shape_bias, summarise
from task1.analysis.feature_similarity import random_pair_baseline, stability, stability_vs_prediction
from task1.analysis.representation import plot_grid, plot_per_backbone, tsne_2d
from task1.data.datasets import load_stl10_arrays, to_common_canvas
from task1.data.make_cue_conflicts import load_accepted
from task1.data.transforms import (DIRECTIONS, grayscale, hue_rotation, make_patch_permutations,
                                   patch_shuffle, translate)
from task1.models.backbones import BACKBONES, build_backbone
from task1.models.linear_head import train_linear_head

MODELS = ["resnet50", "vit_b16", "clip_b32", "clip_zs"]          # clip_zs = zero-shot CLIP
PRETTY = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "clip_b32": "CLIP head", "clip_zs": "CLIP zero-shot"}


# ---------------------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------------------
def condition_list(cfg):
    conds = ["clean", "grayscale", "color_extra", "shuffle"]
    for d in cfg["translation"]["deltas"]:
        if d > 0:
            conds += [f"trans_{name}_{d}" for name in DIRECTIONS]
    return conds


def apply_condition(cond, x, perms, cfg):
    if cond == "clean":
        return x
    if cond == "grayscale":
        return grayscale(x)
    if cond == "color_extra":
        if cfg["color"]["extra"] == "hue_rotation":
            return hue_rotation(x, cfg["color"]["hue_factor"])
        raise NotImplementedError("Add palette transfer / class-swapped statistics here if you choose them.")
    if cond == "shuffle":
        return patch_shuffle(x, perms, cfg["patch_shuffle"]["grid"])
    if cond.startswith("trans_"):
        _, direction, delta = cond.split("_")
        return translate(x, int(delta), direction)
    raise ValueError(cond)


# ---------------------------------------------------------------------------------------
# Feature extraction (the only place backbones run)
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def features_from_canvases(bb, images_u8, device, bs, fn=None):
    out = []
    for s in range(0, len(images_u8), bs):
        x = images_u8[s:s + bs].to(device).float().div_(255.0)
        if fn is not None:
            x = fn(x, slice(s, s + bs))
        out.append(bb(x).cpu())
    return torch.cat(out)


@torch.no_grad()
def features_from_stl(bb, data96, indices, device, bs, size):
    out = []
    for s in range(0, len(indices), bs):
        out.append(bb(to_common_canvas(data96[indices[s:s + bs]], size).to(device)).cpu())
    return torch.cat(out)


def compute_all_features(cfg, device, recompute=False):
    res = Path(resolve(cfg["results_dir"]))
    cache = res / "cache"
    clean = torch.load(cache / "clean_subset.pt")
    images, classes = clean["images"], clean["classes"]

    grid = cfg["patch_shuffle"]["grid"]
    perms = torch.as_tensor(make_patch_permutations(len(images), grid, cfg["seed"]))
    save_json({"seed": cfg["seed"], "grid": grid, "perms": perms.numpy()}, res / "patch_permutations.json")

    accepted, cue_summary = load_accepted(cfg)
    cue_images = torch.load(cache / "cue_conflicts.pt")["images"][torch.as_tensor(accepted.id.values)]

    split = load_json(Path(resolve(cfg["splits_dir"])) / f"stl10_trainval_seed{cfg['seed']}.json")
    x96, _, _ = load_stl10_arrays(resolve(cfg["data_root"]), "train")

    conds = condition_list(cfg)
    for name in BACKBONES:
        path = cache / f"features_{name}.pt"
        if path.exists() and not recompute:
            continue
        print(f"[features] {name}")
        bb = build_backbone(name, device)
        feats = {
            "train": features_from_stl(bb, x96, np.asarray(split["train"]), device, cfg["batch_size"], cfg["image_size"]),
            "val": features_from_stl(bb, x96, np.asarray(split["val"]), device, cfg["batch_size"], cfg["image_size"]),
        }
        for cond in conds:
            fn = lambda x, sl, c=cond: apply_condition(c, x, perms[sl], cfg)   # noqa: E731
            feats[cond] = features_from_canvases(bb, images, device, cfg["batch_size"], fn)
        feats["cue"] = features_from_canvases(bb, cue_images, device, cfg["batch_size"])
        if name == "clip_b32":
            feats["text"] = bb.text_classifier(classes, cfg["zero_shot_template"]).cpu()
            feats["logit_scale"] = float(bb.net.logit_scale.exp())
        torch.save(feats, path)
        del bb
        torch.cuda.empty_cache()
    return accepted, cue_summary, cue_images


# ---------------------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------------------
def main(cfg, recompute=False):
    set_seed(cfg["seed"])
    device = get_device()
    res = Path(resolve(cfg["results_dir"]))
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    accepted, cue_summary, cue_images = compute_all_features(cfg, device, recompute)
    save_json(cue_summary, res / "cue_conflict_counts.json")
    clean = torch.load(res / "cache" / "clean_subset.pt")
    labels, classes = clean["labels"].numpy(), clean["classes"]
    K = len(classes)
    split = load_json(Path(resolve(cfg["splits_dir"])) / f"stl10_trainval_seed{cfg['seed']}.json")
    _, y96, _ = load_stl10_arrays(resolve(cfg["data_root"]), "train")
    ytr = torch.as_tensor(y96[np.asarray(split["train"])])
    yva = torch.as_tensor(y96[np.asarray(split["val"])])
    feats = {n: torch.load(res / "cache" / f"features_{n}.pt") for n in BACKBONES}
    conds = condition_list(cfg) + ["cue"]

    # ---- 1. train the three linear heads (same seed, same recipe) ----------------------
    logits, head_info = {m: {} for m in MODELS}, {}
    for n in BACKBONES:
        head, info = train_linear_head(feats[n]["train"], ytr, feats[n]["val"], yva, K, cfg["head"], cfg["seed"], device)
        head_info[n] = info
        torch.save(head.state_dict(), res / "cache" / f"head_{n}.pt")
        with torch.no_grad():
            for c in conds:
                logits[n][c] = head(feats[n][c].to(device)).cpu()
    # zero-shot CLIP: softmax over logit_scale * cosine(image, prompt)
    for c in conds:
        logits["clip_zs"][c] = feats["clip_b32"]["logit_scale"] * feats["clip_b32"][c] @ feats["clip_b32"]["text"].T
    save_json(head_info, res / "linear_heads.json")
    preds = {m: {c: logits[m][c].argmax(1).numpy() for c in conds} for m in MODELS}

    # ---- 2. clean / colour / shuffle table (absolute AND relative to own clean) ---------
    rows = []
    for m in MODELS:
        base = summarise(logits[m]["clean"], labels, preds[m]["clean"], K)
        for c in [c for c in conds if c != "cue"]:
            r = summarise(logits[m][c], labels, preds[m]["clean"], K)
            r.update(model=PRETTY[m], condition=c, delta_acc=r["acc"] - base["acc"])
            rows.append(r)
    table = pd.DataFrame(rows)[["model", "condition", "acc", "delta_acc", "macro_f1", "mean_max_conf", "consistency"]]
    table.to_csv(res / "table_all_conditions.csv", index=False)
    compact = table[table.condition.isin(["clean", "grayscale", "color_extra", "shuffle"])]
    compact.to_csv(res / "table_clean_color_shuffle.csv", index=False)
    print(compact.round(2).to_string(index=False))

    # ---- 3. translation curve (average over the 4 directions) ---------------------------
    tr_rows = []
    for m in MODELS:
        for d in cfg["translation"]["deltas"]:
            sub = table[(table.model == PRETTY[m]) & (table.condition == "clean")] if d == 0 else \
                table[(table.model == PRETTY[m]) & table.condition.str.fullmatch(rf"trans_[a-z]+_{d}")]
            tr_rows.append({"model": PRETTY[m], "delta": d, "acc": sub.acc.mean(), "consistency": sub.consistency.mean()})
    tr = pd.DataFrame(tr_rows)
    tr.to_csv(res / "table_translation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for m in MODELS:
        s = tr[tr.model == PRETTY[m]]
        axes[0].plot(s.delta, s.acc, marker="o", label=PRETTY[m])
        axes[1].plot(s.delta, s.consistency, marker="o", label=PRETTY[m])
    for ax, t in zip(axes, ["Top-1 accuracy (%)", "Prediction consistency (%)"]):
        ax.set_xlabel("displacement (px)"); ax.set_ylabel(t); ax.set_xticks(cfg["translation"]["deltas"]); ax.grid(alpha=.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "translation_curve.png", dpi=150); plt.close(fig)

    # ---- 4. shape bias + coverage --------------------------------------------------------
    sb_rows, cell_rows = [], []
    for m in MODELS:
        r = shape_bias(preds[m]["cue"], accepted.shape_label, accepted.texture_label)
        r["model"] = PRETTY[m]
        sb_rows.append(r)
        for (pair, direction), g in accepted.groupby(["pair", "direction"]):
            rr = shape_bias(preds[m]["cue"][g.index.values], g.shape_label, g.texture_label)
            rr.update(model=PRETTY[m], pair=pair, direction=direction)
            cell_rows.append(rr)
    pd.DataFrame(sb_rows).to_csv(res / "table_shape_bias.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(res / "table_shape_bias_per_pair.csv", index=False)
    print(pd.DataFrame(sb_rows).round(2).to_string(index=False))

    # ---- 5. representation stability (+ link to prediction changes) ---------------------
    content_pos = torch.as_tensor(accepted.content_pos.values)
    groups = {"grayscale": ["grayscale"], "color_extra": ["color_extra"], "shuffle": ["shuffle"],
              **{f"translation_{d}": [f"trans_{k}_{d}" for k in DIRECTIONS] for d in cfg["translation"]["deltas"] if d > 0},
              "cue_conflict": ["cue"]}
    st_rows = []
    for n in BACKBONES:
        f = feats[n]
        baseline = random_pair_baseline(f["clean"], cfg["seed"])
        for gname, cs in groups.items():
            fc = torch.cat([f["clean"][content_pos] if c == "cue" else f["clean"] for c in cs])
            ft = torch.cat([f[c] for c in cs])
            pc = np.concatenate([preds[n]["clean"][content_pos.numpy()] if c == "cue" else preds[n]["clean"] for c in cs])
            pt = np.concatenate([preds[n][c] for c in cs])
            r = {"backbone": PRETTY[n], "intervention": gname, **stability(fc, ft),
                 "random_pair_baseline": baseline, **stability_vs_prediction(fc, ft, pc, pt)}
            st_rows.append(r)
    st = pd.DataFrame(st_rows)
    st.to_csv(res / "table_representation_stability.csv", index=False)
    print(st.pivot(index="intervention", columns="backbone", values="mean").round(3))

    # ---- 6. CLIP zero-shot vs CLIP head (same features, different decision rule) ----------
    agree = [{"condition": c, "agreement": float((preds["clip_zs"][c] == preds["clip_b32"][c]).mean() * 100)}
             for c in conds]
    pd.DataFrame(agree).to_csv(res / "table_clip_zeroshot_vs_head_agreement.csv", index=False)

    # ---- 7. t-SNE ------------------------------------------------------------------------
    # The handout: "For each backbone, fit ONE two-dimensional projection to the combined
    # clean and transformed features". So each backbone gets a single t-SNE fitted on
    #   clean (500) + grayscale (500) + translation (500) + patch shuffle (500) + cue conflicts
    # colour = ground-truth class (cue conflicts: the content/SHAPE class)
    # marker = condition.
    # Translation has 12 variants (4 directions x 3 displacements); plotting all of them would
    # swamp the map, so ONE representative (largest shift, one direction) is shown. The
    # numbers in the stability table still average over all four directions.
    tcfg = cfg["tsne"]
    shifted = [d for d in cfg["translation"]["deltas"] if d > 0]
    if not shifted:
        raise ValueError("translation.deltas must contain at least one non-zero displacement")
    # Only deltas listed under `translation.deltas` were extracted. If the t-SNE setting asks
    # for one that was not extracted, fall back to the largest one instead of crashing with
    # KeyError('trans_right_32') -- this was the original bug.
    t_delta = tcfg["translation_delta"] if tcfg["translation_delta"] in shifted else max(shifted)
    t_dir = tcfg["translation_direction"] if tcfg["translation_direction"] in DIRECTIONS else next(iter(DIRECTIONS))
    if (t_delta, t_dir) != (tcfg["translation_delta"], tcfg["translation_direction"]):
        print(f"[t-SNE] requested {tcfg['translation_direction']} {tcfg['translation_delta']}px was not "
              f"extracted; using {t_dir} {t_delta}px instead")
    tr_key = f"trans_{t_dir}_{t_delta}"

    # (a) REQUIRED figure: one projection per backbone, all conditions together
    fits = {}
    for n in BACKBONES:
        blocks = [("clean", feats[n]["clean"], labels),
                  ("grayscale", feats[n]["grayscale"], labels),
                  (f"translation {t_delta}px", feats[n][tr_key], labels),
                  ("patch shuffle", feats[n]["shuffle"], labels),
                  ("cue conflict", feats[n]["cue"], accepted.shape_label.values)]
        X = torch.cat([b[1] for b in blocks]).numpy()
        lab = np.concatenate([np.asarray(b[2]) for b in blocks])
        cond = np.concatenate([np.full(len(b[1]), b[0]) for b in blocks])
        fits[PRETTY[n]] = (tsne_2d(X, tcfg, cfg["seed"]), lab, cond)
    settings = {"method": "t-SNE (scikit-learn)", "perplexity": tcfg["perplexity"], "metric": tcfg["metric"],
                "init": tcfg["init"], "learning_rate": "auto", "seed": cfg["seed"],
                "translation_shown": f"{t_dir} {t_delta}px",
                "points_per_condition": {b[0]: len(b[1]) for b in blocks},
                "cue_conflict_colour": "content (shape) class"}
    save_json(settings, res / "tsne_settings.json")          # the handout asks you to REPORT these
    plot_per_backbone(fits, classes, fig_dir / "tsne_per_backbone.png",
                      title=f"t-SNE per backbone (perplexity={tcfg['perplexity']}, metric={tcfg['metric']}, "
                            f"init={tcfg['init']}, seed={cfg['seed']}). Coordinates are NOT comparable across panels.")

    # (b) OPTIONAL supplementary figure: a separate fit per (backbone, intervention).
    #     Easier to read for a single intervention, but it is extra -- (a) is the required one.
    if tcfg.get("per_intervention_grid", True):
        tsne_conds = {"grayscale": "grayscale", "color_extra": "color_extra", "shuffle": "shuffle",
                      f"translate {t_delta}px {t_dir}": tr_key, "cue_conflict": "cue"}
        panels = {}
        for row_name, c in tsne_conds.items():
            for n in BACKBONES:
                if c == "cue":
                    fc, lab_c = feats[n]["clean"][content_pos], labels[content_pos.numpy()]
                    lab_t = accepted.shape_label.values
                else:
                    fc, lab_c, lab_t = feats[n]["clean"], labels, labels
                X = torch.cat([fc, feats[n][c]]).numpy()
                emb = tsne_2d(X, tcfg, cfg["seed"])
                is_t = np.r_[np.zeros(len(fc), bool), np.ones(len(feats[n][c]), bool)]
                panels[(row_name, PRETTY[n])] = (emb, np.r_[lab_c, lab_t], is_t)
        plot_grid(panels, classes, fig_dir / "tsne_supplementary_grid.png",
                  title="Supplementary: one t-SNE per (intervention, backbone)")

    # ---- 8. informative cue-conflict examples --------------------------------------------
    dec = pd.DataFrame({m: [decision_type(p, s, t) for p, s, t in
                            zip(preds[m]["cue"], accepted.shape_label, accepted.texture_label)] for m in MODELS})
    picks = []
    for label, mask in [("all shape", (dec == "shape").all(1)), ("all texture", (dec == "texture").all(1)),
                        ("models disagree", dec.nunique(1) > 1), ("all other", (dec == "other").all(1))]:
        picks += [(label, i) for i in np.where(mask)[0][:3]]
    if picks:
        fig, axes = plt.subplots(1, len(picks), figsize=(2.3 * len(picks), 3.6), squeeze=False)
        for ax, (label, i) in zip(axes[0], picks):
            ax.imshow(cue_images[i].permute(1, 2, 0).numpy()); ax.axis("off")
            lines = [f"[{label}]", f"shape={classes[accepted.shape_label[i]]}", f"tex={classes[accepted.texture_label[i]]}"]
            lines += [f"{PRETTY[m][:9]}: {classes[preds[m]['cue'][i]]} ({dec[m][i][0].upper()})" for m in MODELS]
            ax.set_title("\n".join(lines), fontsize=6.5, loc="left")
        fig.tight_layout(); fig.savefig(fig_dir / "cue_conflict_examples.png", dpi=160); plt.close(fig)
    dec.assign(id=accepted.id, shape=accepted.shape_name, texture=accepted.texture_name).to_csv(
        res / "cue_conflict_decisions.csv", index=False)
    print(f"Done. Results in {res}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1/configs/task1.yaml")
    ap.add_argument("--recompute", action="store_true", help="re-extract cached features")
    args = ap.parse_args()
    main(load_config(args.config), args.recompute)
