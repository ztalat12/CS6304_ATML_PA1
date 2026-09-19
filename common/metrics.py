"""Classification metrics used by all tasks.

* Top-1 accuracy: fraction of correct argmax predictions.
* Macro-F1: F1 computed per class and then averaged with equal weight. Unlike accuracy it
  does not let large classes dominate, which matters for PACS (imbalanced classes) and for
  spotting classes a model has silently given up on.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def classification_metrics(y_true, y_pred, num_classes: int) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(1)
    per_class_acc = np.divide(np.diag(cm), support, out=np.full(num_classes, np.nan), where=support > 0)
    return {
        "acc": float((y_true == y_pred).mean() * 100.0),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100.0),
        "per_class_acc": (per_class_acc * 100.0).tolist(),
        "confusion": cm.tolist(),
    }


def top_confusions(cm, class_names, k: int = 5):
    """Largest off-diagonal entries of a confusion matrix: (true, predicted, count)."""
    cm = np.asarray(cm).copy()
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, cm.shape)
        if cm[i, j] > 0:
            out.append({"true": class_names[i], "pred": class_names[j], "count": int(cm[i, j])})
    return out
