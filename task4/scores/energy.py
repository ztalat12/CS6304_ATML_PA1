"""Energy score (Liu et al., 2020), as an UNKNOWNNESS score.

    u_Energy(x) = - log sum_k exp(z_k(x))          (temperature 1)

What it measures: aggregate evidence over ALL logits - a smooth maximum. When one logit dominates, logsumexp
is close to max z (so Energy is close to MLS); when several classes respond, it adds their evidence together.
It keeps the absolute scale that MSP discards. Computed in float64 with the usual max-subtraction trick.
"""
import numpy as np
from scipy.special import logsumexp


def energy(logits) -> np.ndarray:
    return -logsumexp(np.asarray(logits, dtype=np.float64), axis=1)
