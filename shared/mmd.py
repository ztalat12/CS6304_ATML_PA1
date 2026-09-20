"""Multi-kernel Maximum Mean Discrepancy (shared by DAN in Task 2 and DAN-DG in Task 3,
so that the ONLY difference between them is which data are aligned).

MMD^2(P, Q) = || E_P[phi(x)] - E_Q[phi(y)] ||_H^2
            = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]          (kernel trick: no phi needed)

Kernel: k = sum_{m in {0.5, 1, 2}} exp( -||a-b||^2 / (m * median) ),
where `median` is the median pairwise squared distance in the CURRENT combined batch
(the "median heuristic" adapts the bandwidth to the feature scale, which changes as the
network trains). The median is detached: it sets the scale, it is not optimised.

TWO ESTIMATORS (m samples from each side)
-----------------------------------------
biased  (V-statistic, the DEFAULT, used by every main run in Tasks 2 and 3):
    mean over ALL m*m pairs of K_xx and K_yy, i = j included.
    K_xx.mean() = 3/m + (m-1)/m * [mean of the off-diagonal K_xx]      (k(x,x) = 3 always)
unbiased (U-statistic, opt-in with unbiased=True):
    mean over the m*(m-1) pairs with i != j only.

    biased - unbiased = 6/m - (1/m) * ( offdiag-mean K_xx + offdiag-mean K_yy )

So the difference is NOT just a constant. The 6/m part is a constant offset: even two samples
from the SAME distribution give biased MMD^2 ~ (2/m)(3 - typical k) ~ 0.47 at m = 8 and ~ 0.15
at m = 24, so the logged value can never approach 0. The second part has a gradient: it rewards
high within-domain kernel similarity. Because the median is detached, that gradient keeps
pulling ALL features towards smaller norms (at a fixed bandwidth, shrinking everything raises
every kernel value), while the next batch's median rescales the bandwidth, so the loss never
actually drops and the pull never stops. Its strength is about 1/m: three times larger for
Task 3's 8-vs-8 pairs than for Task 2's 24-vs-24. The unbiased estimate has zero expectation for
identical distributions (it can be negative) and no such systematic pull. Write in the report
which estimator was used.
"""
import torch


def squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ||a||^2 + ||b||^2 - 2 a.b  (no sqrt => well-defined gradient at zero distance)
    return (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1) - 2.0 * a @ b.T).clamp_min(0.0)


def mmd2(x: torch.Tensor, y: torch.Tensor, factors=(0.5, 1.0, 2.0), unbiased: bool = False) -> torch.Tensor:
    z = torch.cat([x, y], 0)
    d2 = squared_distances(z, z)
    n = z.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=z.device)
    median = d2.detach()[off_diag].median().clamp_min(1e-8)     # same bandwidth rule for both estimators
    K = sum(torch.exp(-d2 / (f * median)) for f in factors)
    m = x.shape[0]
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    if not unbiased:
        # biased V-statistic -- unchanged from the version every Task 2 / Task 3 main run used
        return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    # unbiased U-statistic: drop the i = j terms and average over the i != j pairs only
    mx, my = Kxx.shape[0], Kyy.shape[0]
    xx = (Kxx.sum() - Kxx.diagonal().sum()) / (mx * (mx - 1))
    yy = (Kyy.sum() - Kyy.diagonal().sum()) / (my * (my - 1))
    return xx + yy - 2.0 * Kxy.mean()
