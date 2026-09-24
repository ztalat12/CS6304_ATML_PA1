"""Decisions fixed BEFORE any unknown image is seen. task4.freeze copies all of this into the freeze record,
so the record shows the tables, comparisons and failure-selection rule were chosen in advance.

Written on 20 Sep 2026, before any Task 4 model was trained.
"""
from itertools import combinations

POSTHOC = ["msp", "mls", "energy", "mahalanobis"]

# Required evidence 1: the four post-hoc scores on the frozen Vanilla model.
TABLE1 = [("vanilla", s) for s in POSTHOC]
# Required evidence 2: Vanilla vs GCSC vs PROSER with MLS, plus PROSER's placeholder-based detection score.
TABLE2 = [("vanilla", "mls"), ("gcsc", "mls"), ("proser", "mls"), ("proser", "proser")]

# Paired comparisons tested for significance (Holm-corrected within each table x unknown group).
COMPARISONS_T1 = [((("vanilla", a)), ("vanilla", b)) for a, b in combinations(POSTHOC, 2)]
COMPARISONS_T2 = [(("gcsc", "mls"), ("vanilla", "mls")),
                  (("proser", "mls"), ("vanilla", "mls")),
                  (("proser", "proser"), ("vanilla", "mls")),
                  (("proser", "mls"), ("gcsc", "mls")),
                  (("proser", "proser"), ("gcsc", "mls"))]
CSA_COMPARISONS = [("gcsc", "vanilla"), ("proser", "vanilla"), ("proser", "gcsc")]

# Failure analysis (handout: "Using the vanilla MLS threshold, inspect at least three incorrectly accepted near
# unknowns and three incorrectly accepted far unknowns").
FAILURE_MODEL, FAILURE_SCORE = "vanilla", "mls"
N_FAILURES_PER_GROUP = 6
FAILURE_RULE = ("Among unknowns ACCEPTED by Vanilla-MLS (u <= tau), take for every CIFAR-100 class its most "
                "confidently accepted image (lowest u_MLS = largest max logit); order these by u_MLS ascending and "
                f"keep the first {N_FAILURES_PER_GROUP} per group (near, far). One image per class keeps the "
                "examples diverse. If fewer than 3 classes of a group have an accepted image, the selection is "
                "topped up to 3 with the next most confidently accepted images of that group.")

# Two SUPPLEMENTARY views of the same accepted unknowns, each in its own notebook cell and results folder. Whether
# they go into the report is decided later; their selection rules are fixed here, before any result exists.
#   C - both ends of each class: what the most confident failure AND a barely-accepted failure look like.
#   B - a seeded random sample: what a TYPICAL accepted unknown looks like (the main rule shows only extremes).
VIEW_CONFIDENT_BORDERLINE_RULE = (
    "For every CIFAR-100 class with at least one image accepted by Vanilla-MLS: its most confidently accepted image "
    "(lowest u_MLS) and its borderline accepted image (highest u_MLS that is still <= tau, i.e. closest to the "
    "threshold). A class with a single accepted image shows it once. Classes in the handout's order.")
N_RANDOM_PER_GROUP = 6
RANDOM_SEED = 6304
VIEW_RANDOM_RULE = (
    f"A random sample of {N_RANDOM_PER_GROUP} accepted images per group (near, far): the group's accepted images are "
    f"sorted by (u_MLS, CIFAR-100 index), then numpy default_rng({RANDOM_SEED}).choice picks {N_RANDOM_PER_GROUP} "
    "without replacement (all of them if fewer); shown in ascending u_MLS order.")

# NOTE: the "semantically plausible vs surprising" labelling is deliberately NOT pre-registered here. It is a
# judgement call about the images, it changes no metric, and the author reviews it herself. The suggested map
# lives separately in configs/plausibility_map.yaml and is applied by evaluation/apply_plausibility_map.py.


def as_record() -> dict:
    return {
        "table1_rows": [list(r) for r in TABLE1],
        "table2_rows": [list(r) for r in TABLE2],
        "comparisons_table1": [[list(a), list(b)] for a, b in COMPARISONS_T1],
        "comparisons_table2": [[list(a), list(b)] for a, b in COMPARISONS_T2],
        "csa_comparisons": [list(c) for c in CSA_COMPARISONS],
        "failure_analysis": {"model": FAILURE_MODEL, "score": FAILURE_SCORE, "rule": FAILURE_RULE,
                             "n_per_group": N_FAILURES_PER_GROUP},
        "failure_views": {"confident_borderline": {"rule": VIEW_CONFIDENT_BORDERLINE_RULE},
                          "random": {"rule": VIEW_RANDOM_RULE, "n_per_group": N_RANDOM_PER_GROUP,
                                     "seed": RANDOM_SEED}},
    }
