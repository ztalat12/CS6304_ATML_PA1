"""Validation-calibrated rejection threshold (one rule for every model and every score).

Handout: "For each model and score, choose a threshold equal to the 95th percentile of unknownness on the
CIFAR-10 validation set; accept x when u(x) <= tau. This uses known validation data only and aims to accept 95%
of known examples."

    tau = 95th percentile of { u(x) : x in the 5,000 CIFAR-10 validation images }
    accept x  <=>  u(x) <= tau          reject x  <=>  u(x) > tau

np.percentile with its default linear interpolation; by construction about 95% of the validation images are
accepted (exactly 95.00% unless scores tie at tau). How well that transfers to the CIFAR-10 TEST set is reported
as "test acceptance".
"""
import numpy as np

PERCENTILE = 95.0


def fit_threshold(u_val, q: float = PERCENTILE) -> float:
    return float(np.percentile(np.asarray(u_val, dtype=np.float64), q))


def accept(u, tau) -> np.ndarray:
    return np.asarray(u) <= tau
