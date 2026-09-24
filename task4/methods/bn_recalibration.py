"""Re-estimate BatchNorm running statistics from REAL CIFAR-10 training images (post-hoc PROSER variant and the
BatchNorm diagnosis; not used by any of the three pre-registered models).

Why this exists. In train mode, BatchNorm normalises each batch with that batch's own mean/variance and only
*records* a running average for later use in eval mode. The running average is never used while training. PROSER's
data-placeholder pass sends manifold-mixed layer2 features through layer3/layer4 in their own forward call, so the
running statistics of those layers become a blend of real-image and mixed-feature statistics. At evaluation time,
real images are then normalised with statistics that partly describe mixed features. Training accuracy can stay at
99.9% while evaluation accuracy drops and the dummy wins on real images.

recalibrate_bn() resets every BatchNorm layer's running statistics and recomputes them as an exact average
(momentum=None) over `n_batches` batches of real training images, in train mode, without gradients. Nothing but the
BatchNorm buffers changes: no weight, no optimiser state, and no training step is affected.
"""
import torch
import torch.nn as nn


@torch.no_grad()
def recalibrate_bn(model, loader, device, n_batches: int, channels_last: bool = False) -> int:
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                       # cumulative (exact) average over the batches below
    was_training = model.training
    model.train()
    seen = 0
    for k, (x, _) in enumerate(loader):
        if k >= n_batches:
            break
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        model.features(x)                        # backbone only; the heads have no BatchNorm
        seen += len(x)
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    model.train(was_training)
    return seen


def bn_buffers(model) -> dict:
    """{layer name: (running_mean, running_var)} copies, for comparing statistics before/after."""
    return {n: (m.running_mean.detach().clone().cpu(), m.running_var.detach().clone().cpu())
            for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}
