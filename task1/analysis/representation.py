"""t-SNE visualisation of clean vs transformed representations.

REQUIRED FIGURE (plot_per_backbone): for EACH backbone, ONE 2-D t-SNE fitted on the
concatenation [clean ; grayscale ; translation ; patch shuffle ; cue conflict] features,
so every condition lives in the same map.
    colour = ground-truth class (for cue conflicts: the content/shape class)
    marker = condition
SUPPLEMENTARY FIGURE (plot_grid): one fit per (backbone, intervention), clean vs one
transformed condition only.
Reading the plots: t-SNE preserves LOCAL neighbourhoods, not global distances or cluster
sizes. Valid statements: "transformed points stay next to their clean class cluster" or
"transformed points form their own island". Invalid: comparing coordinates or distances
between panels (each panel is a separate fit).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def tsne_2d(X: np.ndarray, cfg: dict, seed: int) -> np.ndarray:
    perplexity = min(cfg["perplexity"], (len(X) - 1) / 3)      # t-SNE requires perplexity < N
    return TSNE(n_components=2, perplexity=perplexity, metric=cfg["metric"], init=cfg["init"],
                learning_rate="auto", random_state=seed).fit_transform(X)


def plot_grid(panels, classes, path, title=""):
    """panels: dict[(row_name, col_name)] -> (emb, labels, is_transformed)"""
    rows = list(dict.fromkeys(r for r, _ in panels))          # insertion order
    cols = list(dict.fromkeys(c for _, c in panels))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.2 * len(cols), 3.8 * len(rows)), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            emb, lab, is_t = panels[(r, c)]
            for flag, marker, alpha in [(False, "o", 0.55), (True, "x", 0.8)]:
                m = is_t == flag
                ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(int(k)) for k in lab[m]],
                           marker=marker, s=9, alpha=alpha, linewidths=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(c)
            if j == 0:
                ax.set_ylabel(r)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(k), label=n) for k, n in enumerate(classes)]
    handles += [plt.Line2D([], [], marker="o", ls="", color="gray", label="clean"),
                plt.Line2D([], [], marker="x", ls="", color="gray", label="transformed")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# marker per condition for the required per-backbone figure (translation key is matched by prefix)
_MARKERS = [("clean", "o"), ("grayscale", "s"), ("translation", "^"), ("patch shuffle", "x"), ("cue conflict", "*")]


def plot_per_backbone(fits, classes, path, title=""):
    """fits: dict[backbone] -> (emb [M,2], labels [M], condition names [M]).

    How to read it: are the non-clean markers sitting INSIDE their class's clean cluster
    (representation barely moved), drifting to its edge, or forming a separate island
    (the intervention created its own region of feature space)? For cue conflicts, a star
    inside its SHAPE-colour cluster means the representation follows shape; a star inside
    the cluster of its texture class would show up as the 'wrong' colour in that cluster.
    """
    names = list(fits)
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 5.2), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, n in zip(axes[0], names):
        emb, lab, cond = fits[n]
        for prefix, marker in _MARKERS:
            m = np.char.startswith(cond.astype(str), prefix)
            if not m.any():
                continue
            clean = prefix == "clean"
            ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(int(k)) for k in lab[m]], marker=marker,
                       s=8 if clean else 14, alpha=0.35 if clean else 0.8, linewidths=0.6)
        ax.set_title(n)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(k), label=c) for k, c in enumerate(classes)]
    handles += [plt.Line2D([], [], marker=mk, ls="", color="gray", label=p) for p, mk in _MARKERS]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
