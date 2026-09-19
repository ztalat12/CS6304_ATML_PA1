"""Class-level analysis: is an aggregate gain hiding class-specific negative transfer?"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from shared.pacs import CLASSES


def row_confusions(cm, cls_idx, k=3):
    """Where do the images of one true class go? (dominant confusions)"""
    row = np.asarray(cm)[cls_idx].astype(float)
    total = row.sum()
    row[cls_idx] = 0
    order = np.argsort(row)[::-1][:k]
    return [{"pred": CLASSES[j], "count": int(row[j]), "share_%": 100 * row[j] / max(total, 1)}
            for j in order if row[j] > 0]


def plot_per_class_delta(per_class: dict, baseline: str, path, title):
    methods = [m for m in per_class if m != baseline]
    base = np.asarray(per_class[baseline])
    x = np.arange(len(CLASSES))
    w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, m in enumerate(methods):
        ax.bar(x + i * w - 0.4 + w / 2, np.asarray(per_class[m]) - base, w, label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel(f"target acc change vs {baseline} (pp)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def failure_grid(root, items, idx, preds_a, preds_b, name_a, name_b, path, title, n=8):
    """Images correct under method A (baseline) but wrong under method B."""
    idx = list(idx)[:n]
    if not idx:
        return
    fig, axes = plt.subplots(1, len(idx), figsize=(1.9 * len(idx), 2.4), squeeze=False)
    for ax, i in zip(axes[0], idx):
        ax.imshow(Image.open(Path(root) / items[i][0]).convert("RGB"))
        ax.axis("off")
        ax.set_title(f"true {CLASSES[items[i][1]]}\n{name_a}: {CLASSES[preds_a[i]]}\n{name_b}: {CLASSES[preds_b[i]]}",
                     fontsize=6.5)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(histories: dict, path):
    """Classification loss, alignment/domain loss, discriminator accuracy, source val F1."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["loss_cls"] for r in h], label=name)
        for key in ("loss_mmd", "loss_dom"):
            if key in h[0]:
                axes[1].plot(ep, [r[key] for r in h], label=f"{name}:{key}")
        if "disc_acc" in h[0]:
            axes[2].plot(ep, [100 * r["disc_acc"] for r in h], label=name)
        axes[3].plot(ep, [r["val_mean_macro_f1"] for r in h], label=name)
    for ax, t in zip(axes, ["source CE loss", "alignment / domain loss", "discriminator acc (%)",
                            "mean source-val macro-F1"]):
        ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    axes[2].axhline(50, color="k", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_study(df, path, title):
    """Controlled study: source performance, domain separability and target accuracy vs setting.
    Read the three panels together: more alignment pressure should LOWER separability; the
    question is what it does to the source (middle) and target (right) panels."""
    x = list(df["setting"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, col, t in zip(axes, ["mean_src_f1", "domain_sep", "target_acc"],
                          ["mean source-val macro-F1 (%)", "domain separability (%)  [50 = chance]",
                           "Sketch accuracy (%)"]):
        ax.plot(x, df[col], marker="o")
        for xi, yi in zip(x, df[col]):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=.3)
    axes[1].axhline(50, color="k", ls=":", lw=0.8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
