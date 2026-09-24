"""PROSER placeholder-based detection score, as an UNKNOWNNESS score.

From the reference implementation (github.com/zhoudw-zdw/CVPR21-Proser, proser_unknown_detection.py,
function valdummy, branch CONF_DeltaP - the branch that is switched on):
    totallogits = [known logits, max over the dummy logits]
    p           = softmax(totallogits / 1024.0)
    conf        = p[dummy] - max_k p[known k]        # larger = more unknown
The reference also loops over a bias added to the dummy logit, but its list is [0] (no bias), so the score is
exactly the one above. The rejection rule is then calibrated like every other score in this task: the threshold
is the 95th percentile of the score on CIFAR-10 VALIDATION images (evaluation/thresholds.py).

    u_PROSER(x) = softmax(s/T)_dummy - max_k softmax(s/T)_k,   s = [z_1..z_10, max_j d_j],  T = 1024

Why T = 1024: at such a high temperature every probability is close to 1/11 and the difference behaves like
(max dummy logit - max known logit) / (11 T). The score therefore asks "does the strongest placeholder beat the
strongest known class, and by how much?", on a scale that does not saturate. Computed in float64.
"""
import numpy as np
from scipy.special import softmax

TEMPERATURE = 1024.0


def proser_detection(logits, dummy_logits, temperature=TEMPERATURE) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    d = np.asarray(dummy_logits, dtype=np.float64).max(axis=1, keepdims=True)
    p = softmax(np.concatenate([z, d], axis=1) / temperature, axis=1)
    return p[:, -1] - p[:, :-1].max(axis=1)
