"""Mahalanobis score (after Lee et al., 2018), as an UNKNOWNNESS score - handout variant.

    u_Mah(x) = min_c (f(x) - mu_c)^T Sigma^{-1} (f(x) - mu_c)

f(x) is the 512-d penultimate feature (after global average pooling, before the linear layer).

Handout: "estimate class means mu_c and one shared diagonal covariance Sigma from unaugmented CIFAR-10 training
features, adding 1e-6 to every diagonal entry."
  * mu_c     = mean feature of the 4,500 training images of class c (45k training split, NO augmentation);
  * Sigma    = diag(sigma^2 + 1e-6), where sigma_d^2 = mean over all 45k training images of (f_d - mu_{y,d})^2,
               i.e. the within-class variance of dimension d pooled over the ten classes (divided by N, the
               maximum-likelihood estimate used by Lee et al.'s empirical covariance);
  * with a diagonal Sigma the distance is sum_d (f_d - mu_{c,d})^2 / Sigma_dd - a per-dimension z-scored
    squared Euclidean distance to the nearest class mean.

What it measures: how far the FEATURE is from every known-class cluster, ignoring the linear classifier. An
input can have a large logit (confident) and still sit far from all class means, or vice versa - that is where
it disagrees with MSP/MLS/Energy.

The statistics are computed once from the training features and saved (results/frozen/maha_<run>.npz); the
freeze record stores their SHA-256, and every later evaluation loads that file rather than re-estimating.
"""
import numpy as np

EPS = 1e-6


def fit(features, labels, num_classes=10):
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    means = np.stack([f[y == c].mean(axis=0) for c in range(num_classes)])
    var = ((f - means[y]) ** 2).mean(axis=0) + EPS                  # pooled within-class variance + 1e-6
    return {"means": means, "var": var}


def mahalanobis(features, stats, chunk=2048) -> np.ndarray:
    f = np.asarray(features, dtype=np.float64)
    mu, inv = stats["means"], 1.0 / stats["var"]
    out = np.empty(len(f))
    for s in range(0, len(f), chunk):
        d = f[s:s + chunk, None, :] - mu[None]                      # (n, C, D)
        out[s:s + chunk] = (d * d * inv).sum(axis=2).min(axis=1)    # nearest class mean
    return out


def save(stats, path):
    np.savez(path, means=stats["means"], var=stats["var"])


def load(path):
    d = np.load(path)
    return {"means": d["means"], "var": d["var"]}
