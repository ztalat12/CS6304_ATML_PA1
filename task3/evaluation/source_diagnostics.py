"""Source-side diagnostics for Task 3 -- they never touch Sketch.

1) SOURCE-DOMAIN SEPARABILITY: 3-way logistic regression (photo/art/cartoon) on frozen
   features of the source VALIDATION sets, balanced, 70/30 split, C=1. Chance = 33.3 %.
2) LOCAL SHARPNESS PROXY:  Delta_sharp = L_val(theta + eps) - L_val(theta),
   eps = 0.05 * grad L_val / ||grad L_val||_2, on a FIXED batch of 32 images per source
   domain (seed 6304), model in eval mode. It is one step along the steepest-ascent
   direction: a local, parameterisation-dependent measure -- not 'global flatness'.
"""
import numpy as np
import torch
import torch.nn.functional as F

from shared.diagnostics import extract, separability
from shared.pacs import SOURCE_DOMAINS, PACSImages, eval_transform


def source_domain_separability(model, root, splits, device, seed=6304):
    feats = [extract(model, root, splits[d]["val"], device)[0] for d in SOURCE_DOMAINS]
    return separability(feats, seed)


def fixed_sharpness_batch(root, splits, per_domain=32, seed=6304):
    rng = np.random.RandomState(seed)
    items = []
    for d in SOURCE_DOMAINS:
        val = splits[d]["val"]
        items += [val[i] for i in rng.choice(len(val), per_domain, replace=False)]
    ds = PACSImages(root, items, eval_transform())
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(xs), torch.as_tensor(ys)


def sharpness_proxy(model, x, y, device, radius=0.05):
    model.eval()
    x, y = x.to(device), y.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params)
    norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(radius * g / (norm + 1e-12))
        loss_pert = F.cross_entropy(model(x), y)
        for p, g in zip(params, grads):
            p.sub_(radius * g / (norm + 1e-12))       # restore
    return float(loss_pert - loss.detach()), float(loss.detach()), float(norm)
