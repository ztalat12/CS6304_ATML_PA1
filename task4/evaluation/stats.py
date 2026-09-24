"""Uncertainty for the Task 4 tables (same tools as Tasks 2-3, adapted to AUROC).

  * AUROC 95% CI and paired AUROC differences: stratified bootstrap. Each replicate resamples, with
    replacement, the 10,000 known test images and, separately, the 800 near and 800 far unknowns; every row
    is re-scored on the SAME resampled images, so differences between rows are paired. B = 2000, seed 6304.
    Two-sided bootstrap p-value for a difference: 2 * min(#(diff* <= 0) + 1, #(diff* >= 0) + 1) / (B + 1),
    capped at 1 (never exactly 0).
  * Rates at the fixed threshold (acceptance / rejection, CSA): Wilson 95% interval (n = 10,000 / 800 / 1,600).
  * Paired differences in CSA or rejection between two models on the same images: exact McNemar test on the
    discordant pairs.
  * Holm correction within each family of comparisons.
"""
import numpy as np
from scipy.stats import binomtest, norm, rankdata

B_DEFAULT = 2000


def wilson(k, n, conf=0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(a_ok, b_ok):
    """a_ok, b_ok: boolean arrays on the same items. Returns (n01, n10, p): n01 = only B right, n10 = only A."""
    a_ok, b_ok = np.asarray(a_ok, bool), np.asarray(b_ok, bool)
    n10 = int((a_ok & ~b_ok).sum())
    n01 = int((~a_ok & b_ok).sum())
    p = 1.0 if n10 + n01 == 0 else binomtest(n10, n10 + n01, 0.5).pvalue
    return n01, n10, float(p)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _auc_from_ranks(u_known, u_unknown):
    n0, n1 = len(u_known), len(u_unknown)
    r = rankdata(np.concatenate([u_known, u_unknown]))
    return (r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def bootstrap_aurocs(rows: dict, u_test_key="test", B=B_DEFAULT, seed=6304):
    """rows = {row_name: {"test": u_test, "near": u_near, "far": u_far}} (all rows on the SAME images).
    Returns {row_name: {group: array of B bootstrap AUROCs (in %)}} for group in near/far/all."""
    rng = np.random.default_rng(seed)
    any_row = next(iter(rows.values()))
    n_t, n_n, n_f = len(any_row["test"]), len(any_row["near"]), len(any_row["far"])
    out = {r: {g: np.empty(B) for g in ("near", "far", "all")} for r in rows}
    for b in range(B):
        it, ine, ifa = rng.integers(0, n_t, n_t), rng.integers(0, n_n, n_n), rng.integers(0, n_f, n_f)
        for r, u in rows.items():
            t, ne, fa = u["test"][it], u["near"][ine], u["far"][ifa]
            out[r]["near"][b] = 100 * _auc_from_ranks(t, ne)
            out[r]["far"][b] = 100 * _auc_from_ranks(t, fa)
            out[r]["all"][b] = 100 * _auc_from_ranks(t, np.concatenate([ne, fa]))
    return out


def ci(samples, conf=0.95):
    lo, hi = np.percentile(samples, [50 * (1 - conf), 100 - 50 * (1 - conf)])
    return float(lo), float(hi)


def boot_p(diff_samples):
    """Two-sided bootstrap p-value with the +1 correction, so it is never reported as exactly 0
    (the smallest possible value is 2 / (B + 1))."""
    d = np.asarray(diff_samples)
    B = len(d)
    return float(min(1.0, 2 * min((d <= 0).sum() + 1, (d >= 0).sum() + 1) / (B + 1)))
