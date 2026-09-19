"""Prediction-level metrics for Task 1.

  accuracy / macro-F1         -- absolute performance under a condition
  mean max confidence         -- average of max softmax probability (how sure the model is)
  prediction consistency      -- fraction of images whose predicted class is UNCHANGED
                                 relative to the model's own clean prediction
                                 (Consistency(d) = 1/N sum 1[yhat(x_i) == yhat(T_d(x_i))]).
                                 Note: consistency ignores correctness -- a model that is
                                 consistently wrong is still 'consistent'.
  shape bias / coverage       -- on cue-conflict images (Geirhos et al., 2019)
"""
import numpy as np
import torch

from common.metrics import classification_metrics


def summarise(logits: torch.Tensor, labels, clean_pred=None, num_classes: int = 10) -> dict:
    probs = logits.softmax(1)
    pred = probs.argmax(1).numpy()
    m = classification_metrics(labels, pred, num_classes)
    out = {"acc": m["acc"], "macro_f1": m["macro_f1"],
           "mean_max_conf": float(probs.max(1).values.mean() * 100.0)}
    if clean_pred is not None:
        out["consistency"] = float((pred == np.asarray(clean_pred)).mean() * 100.0)
    return out


def shape_bias(pred, shape_labels, texture_labels) -> dict:
    """Each decision is 'shape' (= content class), 'texture' (= style class) or 'other'.

    Shape Bias = N_shape / (N_shape + N_texture)      -- ignores 'other' decisions!
    Coverage   = (N_shape + N_texture) / N_total      -- how much evidence SB rests on.
    A 90 % shape bias with 10 % coverage is only ~0.1 * N decisions: weak evidence.
    """
    pred, s, t = map(np.asarray, (pred, shape_labels, texture_labels))
    n_shape, n_tex = int((pred == s).sum()), int((pred == t).sum())
    n_other = len(pred) - n_shape - n_tex
    decided = n_shape + n_tex
    return {"n_shape": n_shape, "n_texture": n_tex, "n_other": n_other, "n_total": len(pred),
            "shape_bias": 100.0 * n_shape / decided if decided else float("nan"),
            "coverage": 100.0 * decided / len(pred)}


def decision_type(pred, shape_label, texture_label) -> str:
    return "shape" if pred == shape_label else ("texture" if pred == texture_label else "other")
