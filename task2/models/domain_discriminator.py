"""Gradient reversal layer (GRL) and domain discriminator for DANN / CDAN.

GRL (Ganin et al., 2016): identity in the forward pass; multiplies the gradient by -alpha
in the backward pass. So with ONE backward pass of  L_cls + L_dom :
  * the discriminator D receives +grad L_dom      -> it gets BETTER at telling domains apart
  * the feature extractor F receives -alpha grad L_dom -> it makes domains HARDER to tell apart
This implements the min-max game  min_F max_D [ L_cls - L_dom ]  without alternating updates.
"""
import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha: float):
    return _GradReverse.apply(x, alpha)


def dann_alpha(progress: float, max_alpha: float = 1.0) -> float:
    """alpha(p) = max_alpha * (2 / (1 + exp(-10 p)) - 1),  p in [0, 1].
    Starts at 0 (let the classifier learn first), saturates near max_alpha."""
    p = torch.tensor(float(progress))
    return float(max_alpha * (2.0 / (1.0 + torch.exp(-10.0 * p)) - 1.0))


class DomainDiscriminator(nn.Module):
    """in_dim -> 256 -> ReLU -> Dropout(0.5) -> 2 logits (source=0, target=1)."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)
