"""DAN-DG: pairwise MMD between the OBSERVED source domains (no target access at all).

  L = L_ERM + (lambda_DG / 3) * sum_{e < e'} MMD^2( F(X_e), F(X_e') )
      pairs: (photo, art), (photo, cartoon), (art, cartoon)

Each pair uses 8 + 8 features from the current batch, and the median bandwidth is
computed per pair (inside `mmd2`), exactly the same kernel code as Task 2's DAN.
Hypothesis being tested: if F ignores what distinguishes photo/art/cartoon, maybe it also
ignores what distinguishes sketch. Risk: marginal alignment can also erase class cues.

mmd_estimator
  "biased"   (default) the Task 2 estimator. Every main and study run uses it, as the handout
             requires ("same MMD implementation as Task 2").
  "unbiased" DIAGNOSTIC ONLY (task3/configs/diag_dan_dg_unbiased.yaml). It tests whether the
             lambda_DG = 1 collapse comes from the biased estimator's small-sample pull
             (see shared/mmd.py). It is never the reported main result.

Extra log: feat_norm = mean L2 norm of the 512-d features in the batch. It is only logged
and never enters the loss. If the shrinking explanation is right, it falls steadily in a
collapsing run and stays roughly flat in a healthy one.
"""
import itertools

import torch.nn.functional as F

from shared.engine import Method
from shared.mmd import mmd2

ESTIMATORS = ("biased", "unbiased")


class DANDG(Method):
    uses_target = False

    def __init__(self, lambda_dg: float = 1.0, mmd_estimator: str = "biased", **_):
        super().__init__()
        if mmd_estimator not in ESTIMATORS:
            raise ValueError(f"mmd_estimator must be one of {ESTIMATORS}, got {mmd_estimator!r}")
        self.lambda_dg = lambda_dg
        self.unbiased = mmd_estimator == "unbiased"
        if self.unbiased:
            print("NOTE: DAN-DG is using the UNBIASED MMD estimator -- diagnostic run, not the main result.")

    def loss(self, model, xs, ys, ds, xt, progress):
        f, logits = model(xs, return_features=True)
        loss_cls = F.cross_entropy(logits, ys)
        pairs = list(itertools.combinations(range(3), 2))
        loss_mmd = sum(mmd2(f[ds == a], f[ds == b], unbiased=self.unbiased) for a, b in pairs) / len(pairs)
        return loss_cls + self.lambda_dg * loss_mmd, {"loss_cls": loss_cls.item(), "loss_mmd": loss_mmd.item(),
                                                      "feat_norm": f.detach().norm(dim=1).mean().item()}
