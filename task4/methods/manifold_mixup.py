"""Manifold mixup for PROSER's data placeholders.

Handout: "For two examples from different classes, let h_i = phi_pre(x_i), h_j = phi_pre(x_j), where phi_pre
denotes the network up to layer2. Sample lambda ~ Beta(2, 2) and construct
        h_tilde = lambda * h_i + (1 - lambda) * h_j,    y_i != y_j."

Mixing happens on the layer2 feature MAPS (128 x 16 x 16), not on pixels: the mixed map is then passed through
layer3, layer4 and the heads. Such a representation lies between two learned class regions, which is where an
unknown input is likely to land - so it serves as a proxy unknown.

Implementation choices (stated in the README):
  * Pairing: a random permutation of the half-batch pairs example i with example perm[i]. Pairs with the SAME
    class are dropped (the handout requires y_i != y_j). With 10 balanced classes about 10% are dropped, so a
    64-image half gives about 57 mixed representations per step.
  * One lambda per mini-batch, as in the reference implementation (`Beta(alpha, alpha).sample([])`).
  * Both random streams have their own seeded generator, so they do not depend on anything else.
"""
import numpy as np
import torch


class MixupSampler:
    def __init__(self, alpha: float = 2.0, seed: int = 6304):
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)           # lambda stream
        self.gen = torch.Generator().manual_seed(seed)   # pairing stream (CPU generator, device independent)

    def sample_lambda(self) -> float:
        return float(self.rng.beta(self.alpha, self.alpha))

    def different_class_pairs(self, y: torch.Tensor):
        """Indices (i, j) with y[i] != y[j]; j is a random permutation of the half-batch."""
        perm = torch.randperm(len(y), generator=self.gen).to(y.device)
        keep = y != y[perm]
        i = torch.nonzero(keep, as_tuple=False).squeeze(1)
        return i, perm[i]


def mix(h: torch.Tensor, i: torch.Tensor, j: torch.Tensor, lam: float) -> torch.Tensor:
    """h_tilde = lambda * h_i + (1 - lambda) * h_j (gradients flow into both h_i and h_j)."""
    return lam * h[i] + (1.0 - lam) * h[j]
