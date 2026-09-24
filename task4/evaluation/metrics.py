"""Open-set metrics.

AUROC (threshold-free). Probability that a randomly chosen unknown gets a HIGHER unknownness score than a
randomly chosen known test image (ties count one half). 0.5 = no separation, 1.0 = perfect. Computed from ranks
(the Mann-Whitney U statistic), which equals sklearn's roc_auc_score with unknown = positive class. Reported
for Known vs Near, Known vs Far and Known vs All unknowns (near + far pooled).

At the validation-calibrated threshold tau (one operating point):
  test acceptance   % of CIFAR-10 TEST images with u <= tau           (target: about 95%)
  near rejection    % of near unknowns with u > tau
  far rejection     % of far unknowns with u > tau
  FPR@95TPR         % of unknowns ACCEPTED (u <= tau) = 100 - rejection. The handout: "report FPR@95TPR as the
                    fraction of unknown examples incorrectly accepted under this convention" - i.e. the 95% TPR
                    is the one fixed on VALIDATION knowns, not re-tuned on the test set.

CSA (closed-set accuracy): top-1 accuracy of the ten KNOWN logits on the CIFAR-10 test set, before rejection.
"""
import numpy as np
from scipy.stats import rankdata

GROUPS = ("near", "far", "all")


def auroc(u_known, u_unknown) -> float:
    u_known, u_unknown = np.asarray(u_known, np.float64), np.asarray(u_unknown, np.float64)
    n0, n1 = len(u_known), len(u_unknown)
    r = rankdata(np.concatenate([u_known, u_unknown]))           # average ranks for ties
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def pct_accepted(u, tau) -> float:
    return float(100.0 * np.mean(np.asarray(u) <= tau))


def csa(logits, labels) -> float:
    return float(100.0 * np.mean(np.asarray(logits).argmax(1) == np.asarray(labels)))


def split_unknowns(u_unknown, group):
    """{'near': ..., 'far': ..., 'all': ...} score arrays from the unknown set's group labels."""
    group = np.asarray(group)
    return {"near": u_unknown[group == "near"], "far": u_unknown[group == "far"], "all": u_unknown}


def osr_metrics(u_test, u_unknown, group, tau) -> dict:
    parts = split_unknowns(u_unknown, group)
    row = {f"auroc_{g}": 100.0 * auroc(u_test, parts[g]) for g in GROUPS}
    row["test_accept"] = pct_accepted(u_test, tau)
    for g in GROUPS:
        acc = pct_accepted(parts[g], tau)
        row[f"reject_{g}"] = 100.0 - acc
        row[f"fpr95_{g}"] = acc
    return row
