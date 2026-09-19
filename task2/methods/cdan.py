"""CDAN: class-conditional adversarial alignment (Long et al., 2018).

The discriminator sees g(x) = vec(f (x) p): the outer product of the 512-d feature and the
7-d softmax prediction -> 3584-d. Intuitively, the feature is copied into the 'slot' of
the class the classifier believes in, so the discriminator must compare source-dog-like
regions with target-dog-like regions, instead of the two clouds as a whole (DANN).

As required: no entropy conditioning, and neither f nor p is detached. Consequence worth
discussing: the reversed gradient also reaches the classifier head through p, nudging
target predictions toward ones that make domains indistinguishable. (The official CDAN
code detaches p; the handout deliberately does not.)
"""
import torch

from task2.methods.dann import DANN


class CDAN(DANN):
    def __init__(self, feat_dim=512, num_classes=7, hidden=256, dropout=0.5, max_alpha=1.0,
                 disc_input_norm="none", adv_loss="grl", **_):
        super().__init__(feat_dim * num_classes, hidden, dropout, max_alpha, disc_input_norm, adv_loss)

    def disc_input(self, f, logits):
        p = logits.softmax(1)                               # (B, 7)   NOT detached
        return torch.bmm(p.unsqueeze(2), f.unsqueeze(1)).flatten(1)   # (B, 7*512)
