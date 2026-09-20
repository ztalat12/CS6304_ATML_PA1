"""DAN-style alignment:  L = L_cls(source) + lambda * MMD^2( F(x_s), F(x_t) ).

The penalty compares the WHOLE source feature distribution with the WHOLE target one
(marginal alignment); it never knows which target image belongs to which class, so it can
in principle match 'source dogs' with 'target houses' if that lowers the discrepancy.

mmd_estimator: "biased" (default; every Task 2 result was produced with it) or "unbiased".
The option exists only so that, if the TA asks for one estimator across Tasks 2 and 3, the
DAN runs can be repeated with the unbiased one without touching any other code. See
shared/mmd.py for the difference between the two.
"""
import torch
import torch.nn.functional as F

from shared.engine import Method
from shared.mmd import mmd2


class DAN(Method):
    uses_target = True

    def __init__(self, lambda_mmd: float = 1.0, mmd_estimator: str = "biased", **_):
        super().__init__()
        if mmd_estimator not in ("biased", "unbiased"):
            raise ValueError(f"mmd_estimator must be 'biased' or 'unbiased', got {mmd_estimator!r}")
        self.lambda_mmd = lambda_mmd
        self.unbiased = mmd_estimator == "unbiased"

    def loss(self, model, xs, ys, ds, xt, progress):
        n = xs.shape[0]
        f, logits = model(torch.cat([xs, xt]), return_features=True)   # BN frozen -> batch mixing is harmless
        loss_cls = F.cross_entropy(logits[:n], ys)                     # ONLY source labels
        loss_mmd = mmd2(f[:n], f[n:], unbiased=self.unbiased)
        return loss_cls + self.lambda_mmd * loss_mmd, {"loss_cls": loss_cls.item(), "loss_mmd": loss_mmd.item()}
