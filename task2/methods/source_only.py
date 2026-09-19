"""Source-only ERM: cross-entropy on the domain-balanced source batch; target unused.
This checkpoint is ALSO the Task 3 ERM baseline (never retrained)."""
import torch.nn.functional as F

from shared.engine import Method


class SourceOnly(Method):
    uses_target = False

    def loss(self, model, xs, ys, ds, xt, progress):
        loss = F.cross_entropy(model(xs), ys)
        return loss, {"loss_cls": loss.item()}
