"""MLS - Maximum Logit Score (Vaze et al., 2022), as an UNKNOWNNESS score.

    u_MLS(x) = - max_k z_k(x)

What it measures: the ABSOLUTE size of the strongest class response, before softmax normalisation. Two inputs
with identical logit gaps (identical MSP) can differ in MLS if one activates the class evidence more strongly.
Vaze et al. argue that a good closed-set classifier produces large maximum logits on known classes and smaller
ones on unknowns, so MLS keeps information that MSP normalises away.

For PROSER, MLS is computed on the ten KNOWN logits only (the dummy is not a class), as the handout requires.
"""
import numpy as np


def mls(logits) -> np.ndarray:
    return -np.asarray(logits, dtype=np.float64).max(axis=1)
