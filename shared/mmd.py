"""Multi-kernel Maximum Mean Discrepancy (shared by DAN in Task 2 and DAN-DG in Task 3,
so that the ONLY difference between them is which data are aligned).

MMD^2(P, Q) = || E_P[phi(x)] - E_Q[phi(y)] ||_H^2
            = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]          (kernel trick: no phi needed)

Kernel: k = sum_{m in {0.5, 1, 2}} exp( -||a-b||^2 / (m * median) ),
where `median` is the median pairwise squared distance in the CURRENT combined batch
(the "median heuristic" adapts the bandwidth to the feature scale, which changes as the
network trains). The median is detached: it sets the scale, it is not optimised.

Estimator: the biased (V-statistic) estimate using all pairs including i=j. It is always
>= 0 and low-variance for small batches; say which estimator you used in the report.
"""
import torch


def squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ||a||^2 + ||b||^2 - 2 a.b  (no sqrt => well-defined gradient at zero distance)
    return (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1) - 2.0 * a @ b.T).clamp_min(0.0)


def mmd2(x: torch.Tensor, y: torch.Tensor, factors=(0.5, 1.0, 2.0)) -> torch.Tensor:
    z = torch.cat([x, y], 0)
    d2 = squared_distances(z, z)
    n = z.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=z.device)
    median = d2.detach()[off_diag].median().clamp_min(1e-8)
    K = sum(torch.exp(-d2 / (f * median)) for f in factors)
    m = x.shape[0]
    return K[:m, :m].mean() + K[m:, m:].mean() - 2.0 * K[:m, m:].mean()
