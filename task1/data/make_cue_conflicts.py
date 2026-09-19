"""Generate shape-vs-texture cue-conflict images with AdaIN.

For every unordered class pair {A, B} we generate BOTH directions:
    content (shape) = A, style (texture) = B     and     content = B, style = A
Content and style images come from the fixed 500-image Task 1 test subset.

REJECTION RULE -- defined here, BEFORE any model is evaluated, and using only pixels
(never model predictions). An image is auto-accepted iff all three hold:
  (R1) SHAPE KEPT      corr(Sobel edges(content), Sobel edges(stylised)) >= edge_corr_min
  (R2) TEXTURE MOVED   style_gain = 1 - d(stylised, style) / d(content, style) >= style_gain_min
                       where d = L2 distance between multi-level VGG channel mean/std
                       statistics (the texture descriptor AdaIN itself manipulates)
  (R3) NOT DEGENERATE  pixel std of the stylised image >= min_pixel_std
Then open the contact sheets and fill the `manual_accept` column in review.csv (1/0)
to override any case where the automatic rule is visibly wrong (object unrecognisable,
no visible texture, heavy artefacts). Record accepted/rejected counts in the report.

Run:
  python -m task1.data.make_cue_conflicts --config task1/configs/task1.yaml
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common.io import load_config, resolve, save_json
from common.seed import get_device, set_seed
from task1.data.adain import AdaIN
from task1.data.datasets import canvas_to_uint8

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)


def edge_map(x01):
    g = (x01 * torch.tensor([0.299, 0.587, 0.114], device=x01.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    g = F.avg_pool2d(g, 3, 1, 1)                       # light blur: ignore fine texture edges
    kx = _SOBEL_X.to(x01.device)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, kx.transpose(2, 3), padding=1)
    return (gx ** 2 + gy ** 2).sqrt().flatten(1)


def pearson(a, b):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def main(cfg):
    set_seed(cfg["seed"])
    device = get_device()
    cc = cfg["cue_conflict"]
    res = Path(resolve(cfg["results_dir"]))
    out = res / "cue_conflicts"
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    clean = torch.load(res / "cache" / "clean_subset.pt")
    images, labels, classes = clean["images"], clean["labels"].numpy(), clean["classes"]
    name_to_id = {c: i for i, c in enumerate(classes)}
    net = AdaIN(resolve(cc["vgg"]), resolve(cc["decoder"])).to(device)
    rng = np.random.RandomState(cfg["seed"])

    rows, stylised = [], []
    for pair_id, (a_name, b_name) in enumerate(cc["pairs"]):
        a, b = name_to_id[a_name], name_to_id[b_name]
        for shape_c, tex_c in [(a, b), (b, a)]:                     # both directions
            content_pos = rng.choice(np.where(labels == shape_c)[0], cc["per_direction"], replace=False)
            style_pos = rng.choice(np.where(labels == tex_c)[0], cc["per_direction"], replace=False)
            for cp, sp in zip(content_pos, style_pos):
                c = images[cp:cp + 1].float().div(255).to(device)
                s = images[sp:sp + 1].float().div(255).to(device)
                y = net(c, s, alpha=cc["alpha"])
                # ---- pre-registered, model-free quality measurements ----
                edge_corr = pearson(edge_map(c), edge_map(y)).item()
                sc, ss, sy = net.multi_level_stats(c), net.multi_level_stats(s), net.multi_level_stats(y)
                style_gain = 1.0 - ((sy - ss).norm() / ((sc - ss).norm() + 1e-8)).item()
                pix_std = y.std().item()
                auto = int(edge_corr >= cc["edge_corr_min"] and style_gain >= cc["style_gain_min"]
                           and pix_std >= cc["min_pixel_std"])
                rows.append({"id": len(rows), "pair_id": pair_id, "pair": f"{a_name}-{b_name}",
                             "direction": f"{classes[shape_c]}->{classes[tex_c]}",
                             "content_pos": int(cp), "style_pos": int(sp),
                             "shape_label": int(shape_c), "texture_label": int(tex_c),
                             "shape_name": classes[shape_c], "texture_name": classes[tex_c],
                             "edge_corr": edge_corr, "style_gain": style_gain, "pixel_std": pix_std,
                             "auto_accept": auto, "manual_accept": ""})
                stylised.append(canvas_to_uint8(y.cpu()))

    stylised = torch.cat(stylised)
    df = pd.DataFrame(rows)
    torch.save({"images": stylised}, res / "cache" / "cue_conflicts.pt")
    review_path = out / "review.csv"
    if review_path.exists():
        print(f"{review_path} exists -- NOT overwriting your manual review. Delete it to regenerate.")
    else:
        df.to_csv(review_path, index=False)

    # ---- contact sheets for manual review: content | style | stylised ----
    for (pair, direction), g in df.groupby(["pair", "direction"]):
        n = len(g)
        fig, axes = plt.subplots(3, n, figsize=(1.3 * n, 4.2))
        for j, (_, r) in enumerate(g.iterrows()):
            for i, img in enumerate([images[r.content_pos], images[r.style_pos], stylised[r.id]]):
                axes[i, j].imshow(img.permute(1, 2, 0).numpy())
                axes[i, j].axis("off")
            axes[0, j].set_title(f"#{r.id}\n{'OK' if r.auto_accept else 'REJ'}", fontsize=6)
        fig.suptitle(f"{direction}  (rows: content, style, stylised)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "sheets" / f"{direction.replace('->', '_to_')}.png", dpi=110)
        plt.close(fig)

    save_json({"generated": len(df), "auto_accepted": int(df.auto_accept.sum()),
               "auto_rejected": int((1 - df.auto_accept).sum()), "rule": {k: cc[k] for k in
               ["edge_corr_min", "style_gain_min", "min_pixel_std", "alpha"]}},
              out / "generation_summary.json")
    print(df.groupby("direction").auto_accept.agg(["sum", "count"]))


def load_accepted(cfg):
    """Final accepted set = manual_accept where filled in, else auto_accept.
    Optionally trims to an equal count per (pair, direction) cell for balance."""
    res = Path(resolve(cfg["results_dir"]))
    df = pd.read_csv(res / "cue_conflicts" / "review.csv", keep_default_na=False)
    manual = pd.to_numeric(df.manual_accept, errors="coerce")        # blank -> NaN
    df["accept"] = np.where(manual.isin([0, 1]), manual.fillna(0), df.auto_accept).astype(int)
    acc = df[df.accept == 1]
    counts = acc.groupby(["pair", "direction"]).size()
    balanced = False
    if cfg["cue_conflict"].get("balance", True) and counts.min() * len(counts) >= 200:
        m = counts.min()
        acc = acc.groupby(["pair", "direction"]).head(m)     # keep the first m ids per cell
        balanced = True
    summary = {"generated": len(df), "accepted_before_balance": int(df.accept.sum()),
               "rejected": int((df.accept == 0).sum()), "used": len(acc), "balanced": balanced,
               "per_cell_used": {f"{k[0]}|{k[1]}": int(v) for k, v in
                                 acc.groupby(["pair", "direction"]).size().items()}}
    if len(acc) < 200:
        print(f"WARNING: only {len(acc)} valid conflicts (< 200). Increase per_direction and regenerate.")
    return acc.sort_values("id").reset_index(drop=True), summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1/configs/task1.yaml")
    main(load_config(ap.parse_args().config))
