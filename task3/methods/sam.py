"""Sharpness-Aware Minimisation (Foret et al., 2021), non-adaptive, with AdamW as base.

Objective:  min_theta  max_{||eps||_2 <= rho}  L_ERM(theta + eps)
First-order solution of the inner max:  eps_hat = rho * g / ||g||_2,  g = grad L(theta).

One update = two forward/backward passes on the SAME augmented batch:
  1. g = grad L(theta);  theta <- theta + eps_hat            (climb to the worst nearby point)
  2. g_sam = grad L(theta + eps_hat);  theta <- theta - eps_hat (go back)
  3. AdamW step using g_sam                                   (descend with the 'sharp' gradient)
Frozen BN running statistics (shared policy) apply to both passes, so the perturbed pass
cannot pollute BN statistics.
"""
import torch
import torch.nn.functional as F

from shared.engine import Method


class SAM(Method):
    uses_target = False

    def __init__(self, rho: float = 0.05, **_):
        super().__init__()
        self.rho = rho

    def training_step(self, model, optimizer, xs, ys, ds, xt, progress):
        params = [p for p in model.parameters() if p.requires_grad]
        # ---- pass 1: gradient at theta ----
        loss = F.cross_entropy(model(xs), ys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
        scale = self.rho / (grad_norm + 1e-12)
        eps = []
        with torch.no_grad():
            for p in params:
                e = None if p.grad is None else p.grad * scale
                if e is not None:
                    p.add_(e)                          # theta + eps_hat
                eps.append(e)
        # ---- pass 2: gradient at theta + eps_hat ----
        optimizer.zero_grad(set_to_none=True)
        loss_adv = F.cross_entropy(model(xs), ys)
        loss_adv.backward()
        with torch.no_grad():
            for p, e in zip(params, eps):
                if e is not None:
                    p.sub_(e)                          # restore theta (grads stay = SAM gradient)
        gn = self.clip_()                              # same clipping rule as every other method
        optimizer.step()                               # AdamW step at theta with g_sam
        # loss_cls  : loss at theta            (comparable with ERM/DAN-DG)
        # loss_sam  : loss at theta + eps_hat  (always >= loss_cls; the gap is the local sharpness seen during training)
        # ascent_grad_norm : ||grad L(theta)||, the norm used to build eps_hat
        return {"loss_cls": loss.item(), "loss_sam": loss_adv.item(),
                "ascent_grad_norm": grad_norm.item(), "grad_norm": gn}
