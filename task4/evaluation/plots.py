"""Figures for Task 4.

fig_vanilla_scores - the handout's "one compact multi-panel figure" for MSP, MLS and Mahalanobis on the frozen
Vanilla model. Top row: score distributions of known test images, near and far unknowns, with the frozen
threshold tau (dashed). Bottom row: ROC curves Known vs Near and Known vs Far, with the operating point that tau
produces (unknown = positive class: TPR = % unknowns rejected, FPR = % known test images rejected).
MSP is drawn on a log10 axis because almost all known images have 1 - max p below 1e-3.

fig_acceptance_by_class - % of each CIFAR-100 class accepted as known, for every Table 2 row.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from task4.data.cifar100_unknowns import FAR, NEAR
from task4.evaluation.metrics import auroc

COLORS = {"known": "#1f77b4", "near": "#d62728", "far": "#2ca02c"}
PANELS = [("msp", "MSP: 1 - max softmax", True), ("mls", "MLS: - max logit", False),
          ("mahalanobis", "Mahalanobis: min class distance", False)]


def _save(fig, stem):
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def score_figure(Uv, group, taus, stem):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for col, (s, title, logx) in enumerate(PANELS):
        known, near, far = Uv["test"][s], Uv["unknown"][s][group == "near"], Uv["unknown"][s][group == "far"]
        tau = taus[s]
        tr = (lambda v: np.log10(np.maximum(v, 1e-12))) if logx else (lambda v: v)
        allv = tr(np.concatenate([known, near, far]))
        lo, hi = np.percentile(allv, [0.2, 99.8])
        bins = np.linspace(lo, hi, 70)
        ax = axes[0, col]
        for name, v in (("known", known), ("near", near), ("far", far)):
            ax.hist(np.clip(tr(v), lo, hi), bins=bins, density=True, histtype="step", lw=1.6, color=COLORS[name],
                    label=f"{'CIFAR-10 test' if name == 'known' else name + ' unknown'} (n={len(v)})")
        ax.axvline(tr(np.array([tau]))[0], ls="--", c="k", lw=1.2, label="tau (val 95th pct)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("log10 u" if logx else "u  (larger = more unknown)")
        ax.set_ylabel("density")
        if col == 0:
            ax.legend(fontsize=7.5, loc="upper left")

        ax = axes[1, col]
        for name, v in (("near", near), ("far", far)):
            y = np.r_[np.zeros(len(known)), np.ones(len(v))]
            fpr, tpr, _ = roc_curve(y, np.r_[known, v])
            ax.plot(100 * fpr, 100 * tpr, c=COLORS[name], lw=1.6,
                    label=f"Known vs {name}: AUROC {100 * auroc(known, v):.1f}")
            ax.plot(100 * np.mean(known > tau), 100 * np.mean(v > tau), "o", c=COLORS[name], ms=6, mec="k")
        ax.plot([], [], "o", c="white", mec="k", ms=6, label="operating point at tau")
        ax.plot([0, 100], [0, 100], ":", c="grey", lw=1)
        ax.set_xlabel("% known test images rejected (FPR)")
        ax.set_ylabel("% unknowns rejected (TPR)")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    fig.suptitle("Frozen Vanilla ResNet-18: CIFAR-10 test (known) vs CIFAR-100 near / far unknowns", fontsize=12)
    fig.tight_layout()
    _save(fig, stem)


def acceptance_by_class_figure(pc, rows, nice, stem):
    classes = NEAR + FAR
    fig, ax = plt.subplots(figsize=(13, 4))
    w = 0.8 / len(rows)
    for k, (run, s) in enumerate(rows):
        sub = pc[(pc["model"] == run) & (pc["score"] == nice[s])].set_index("class").loc[classes]
        ax.bar(np.arange(len(classes)) + (k - (len(rows) - 1) / 2) * w, sub["accepted_%"], width=w,
               label=f"{run} - {nice[s]}")
    ax.axvline(len(NEAR) - 0.5, c="k", lw=0.8)
    ax.text(len(NEAR) / 2 - 0.5, 103, "near unknowns", ha="center", fontsize=9)
    ax.text(len(NEAR) + len(FAR) / 2 - 0.5, 103, "far unknowns", ha="center", fontsize=9)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_ylabel("% accepted as known (u <= tau)")
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.6, len(classes) - 0.4)
    ax.legend(fontsize=8.5, ncol=len(rows), loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    _save(fig, stem)


def failure_grid(images, titles, stem, suptitle, row_labels=None):
    """images: list of rows (lists) of 32x32x3 uint8 arrays, or None for an empty slot (its title is still shown);
    titles: same shape; row_labels: optional text written to the left of each row."""
    n_rows, n_cols = len(images), max(1, max(len(r) for r in images))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.55 * n_rows), squeeze=False)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(images[i]):
                if images[i][j] is not None:
                    ax.imshow(images[i][j], interpolation="nearest")
                else:
                    ax.imshow(np.full((32, 32, 3), 235, np.uint8))
                ax.set_title(titles[i][j], fontsize=7.5)
        if row_labels:
            axes[i, 0].text(-0.12, 0.5, row_labels[i], transform=axes[i, 0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=9, fontweight="bold")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    _save(fig, stem)
