"""MSP - Maximum Softmax Probability (Hendrycks & Gimpel, 2017), as an UNKNOWNNESS score.

    u_MSP(x) = 1 - max_k p_k(x),      p_k(x) = exp(z_k) / sum_j exp(z_j)

What it measures: NORMALISED confidence. Softmax divides by the sum over classes, so only the gaps between
logits matter - adding the same constant to every logit leaves MSP unchanged. A strongly activated input and a
weakly activated input with the same logit gaps get the same score.

Numerics: a well-trained network gives max p = 1 - 1e-9 on many images, and 1 - p computed naively in float32
rounds to exactly 0, creating thousands of ties. We compute the same quantity exactly:
    let e_k = exp(z_k - max z), S = sum_k e_k (so max p = 1/S)
    1 - max p = (S - 1)/S = r / (1 + r),  where r = sum of e_k over every class except the arg-max
(r is summed directly, never formed as S - 1, so nothing cancels). Same definition, no artificial ties.
"""
import numpy as np


def msp(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    r = np.sort(e, axis=1)[:, :-1].sum(axis=1)       # every class except the largest
    return r / (1.0 + r)
