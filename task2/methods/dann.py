"""DANN: adversarial marginal alignment with a gradient-reversal layer.

  L = CE(C(F(x_s)), y_s) + CE(D(GRL_alpha(F(x))), domain)     (unit weight on the domain loss)
Only source images enter the class loss; source AND target enter the domain loss.
Logged diagnostics: discriminator accuracy (near 50 % = confused, OR undertrained, OR
collapsed features -- check it together with loss_cls and source accuracy) and alpha.
"""
import math

import torch
import torch.nn.functional as F

from shared.engine import Method
from task2.models.domain_discriminator import DomainDiscriminator, dann_alpha, grad_reverse


class DANN(Method):
    uses_target = True

    def __init__(self, feat_dim=512, hidden=256, dropout=0.5, max_alpha=1.0,
                 disc_input_norm="none", adv_loss="grl", **_):
        super().__init__()
        self.disc = DomainDiscriminator(feat_dim, hidden, dropout)
        self.max_alpha = max_alpha
        # --- DIAGNOSTIC OPTIONS (both default to the handout's behaviour) -------------------
        # disc_input_norm="l2": scale the discriminator's input to a fixed length. The handout's
        #   frozen-BatchNorm policy means nothing renormalises activations, so the cheapest way for
        #   the backbone to raise an unbounded domain loss is to inflate ||f||. Normalising removes
        #   that route: scaling f no longer changes D's output at all.
        # adv_loss="flip": replace "maximise CE(D(f), true domain)" -- unbounded above, so its
        #   gradient never vanishes -- with the non-saturating GAN objective "minimise
        #   CE(D(f), FLIPPED domain)", which is bounded below by 0 and does have a stationary point.
        #   Implemented with the usual two-step update instead of a gradient-reversal layer.
        self.disc_input_norm = disc_input_norm
        self.adv_loss = adv_loss

    def disc_input(self, f, logits):
        return f                                           # marginal: features only

    def _norm(self, x):
        if self.disc_input_norm == "l2":                   # unit length, rescaled so entries stay O(1)
            return F.normalize(x, dim=1) * math.sqrt(x.shape[1])
        return x

    def loss(self, model, xs, ys, ds, xt, progress):
        n, m = xs.shape[0], xt.shape[0]
        f, logits = model(torch.cat([xs, xt]), return_features=True)
        loss_cls = F.cross_entropy(logits[:n], ys)
        alpha = dann_alpha(progress, self.max_alpha)
        d_logits = self.disc(grad_reverse(self._norm(self.disc_input(f, logits)), alpha))
        d_labels = torch.cat([torch.zeros(n), torch.ones(m)]).long().to(xs.device)
        loss_dom = F.cross_entropy(d_logits, d_labels)
        disc_acc = (d_logits.argmax(1) == d_labels).float().mean().item()
        return loss_cls + loss_dom, {"loss_cls": loss_cls.item(), "loss_dom": loss_dom.item(),
                                     "disc_acc": disc_acc, "alpha": alpha,
                                     "feat_norm": f.norm(dim=1).mean().item()}

    def training_step(self, model, optimizer, xs, ys, ds, xt, progress):
        """Default path = the handout's single backward pass through the GRL.
        adv_loss="flip" instead alternates, as ADDA and most GAN code do:
          pass A  backbone (+ head): minimise  L_cls + alpha * CE(D(f), FLIPPED labels), D frozen
          pass B  discriminator     : minimise CE(D(f.detach()), true labels), backbone frozen
        Same game, but the backbone now minimises something bounded instead of maximising
        something unbounded."""
        if self.adv_loss != "flip":
            return super().training_step(model, optimizer, xs, ys, ds, xt, progress)

        n, m = xs.shape[0], xt.shape[0]
        alpha = dann_alpha(progress, self.max_alpha)
        optimizer.zero_grad(set_to_none=True)

        for prm in self.disc.parameters():
            prm.requires_grad_(False)                      # pass A: D is a fixed critic
        f, logits = model(torch.cat([xs, xt]), return_features=True)
        loss_cls = F.cross_entropy(logits[:n], ys)
        d_in = self._norm(self.disc_input(f, logits))
        flipped = torch.cat([torch.ones(n), torch.zeros(m)]).long().to(xs.device)
        loss_g = F.cross_entropy(self.disc(d_in), flipped)
        (loss_cls + alpha * loss_g).backward()
        for prm in self.disc.parameters():
            prm.requires_grad_(True)

        d_labels = torch.cat([torch.zeros(n), torch.ones(m)]).long().to(xs.device)
        d_logits = self.disc(d_in.detach())                # pass B: backbone frozen by detaching
        loss_d = F.cross_entropy(d_logits, d_labels)
        loss_d.backward()

        gn = self.clip_()
        optimizer.step()
        return {"loss_cls": loss_cls.item(), "loss_dom": loss_d.item(), "loss_gen": loss_g.item(),
                "disc_acc": (d_logits.argmax(1) == d_labels).float().mean().item(), "alpha": alpha,
                "feat_norm": f.norm(dim=1).mean().item(), "grad_norm": gn,
                "loss_total": loss_cls.item() + alpha * loss_g.item()}
