"""Unknownness scores. Every score maps saved model outputs to u(x); LARGER u = MORE likely unknown.

All scores of one model read exactly the same saved arrays (cache/<run>/<split>.npz), as the handout requires:
"All four scores must use exactly the same saved logits and features."

    msp          1 - max_k softmax(z)_k                         logits
    mls          - max_k z_k                                    logits
    energy       - log sum_k exp(z_k)                           logits
    mahalanobis  min_c (f - mu_c)^T Sigma^-1 (f - mu_c)         features + training statistics
    proser       softmax(s/1024)_dummy - max_k softmax(s/1024)_k,  s = [z, max dummy]    PROSER only

For PROSER, msp/mls/energy use the ten known logits only.
"""
from task4.scores import mahalanobis as maha
from task4.scores.energy import energy
from task4.scores.mls import mls
from task4.scores.msp import msp
from task4.scores.proser_detection import TEMPERATURE, proser_detection

DEFINITIONS = {
    "msp": "u = 1 - max_k softmax(z)_k  (10 known logits; computed exactly in float64)",
    "mls": "u = - max_k z_k  (10 known logits)",
    "energy": "u = - logsumexp_k z_k  (10 known logits, T = 1)",
    "mahalanobis": "u = min_c sum_d (f_d - mu_cd)^2 / (sigma_d^2 + 1e-6); mu_c, sigma^2 = class means and pooled "
                   "within-class variance of UNAUGMENTED 45k CIFAR-10 training features",
    "proser": f"u = p_dummy - max_k p_k, p = softmax([z, max_j d_j] / {TEMPERATURE:g})  (PROSER reference code)",
}
POSTHOC = ["msp", "mls", "energy", "mahalanobis"]


def compute(out: dict, maha_stats=None, temperature=TEMPERATURE) -> dict:
    """out = one loaded cache file (logits, features, optional dummy_logits) -> {score name: u array}."""
    u = {"msp": msp(out["logits"]), "mls": mls(out["logits"]), "energy": energy(out["logits"])}
    if maha_stats is not None:
        u["mahalanobis"] = maha.mahalanobis(out["features"], maha_stats)
    if out.get("dummy_logits") is not None:
        u["proser"] = proser_detection(out["logits"], out["dummy_logits"], temperature)
    return u
