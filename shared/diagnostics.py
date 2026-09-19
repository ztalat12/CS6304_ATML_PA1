"""Diagnostics shared by Tasks 2 and 3: feature extraction and domain separability.

DOMAIN SEPARABILITY = held-out accuracy of a logistic-regression probe that predicts the
domain from frozen features. 50 % (binary) or 33.3 % (3-way) = chance.
It measures how much DOMAIN information is still linearly recoverable. It says nothing
about whether CLASS information survived -- always read it next to accuracy/macro-F1.
Features are standardised (fit on the probe's training part only) because logistic
regression is scale-sensitive; C = 1 as required.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from shared.pacs import PACSImages, eval_transform


@torch.no_grad()
def extract(model, root, items, device, batch_size=128, num_workers=2):
    """Returns (features, logits, labels) with the model in FULL eval mode."""
    model.eval()
    loader = DataLoader(PACSImages(root, items, eval_transform()), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    F, Z, Y = [], [], []
    for x, y in loader:
        f, z = model(x.to(device), return_features=True)
        F.append(f.cpu()); Z.append(z.cpu()); Y.append(y)
    return torch.cat(F).numpy(), torch.cat(Z).numpy(), torch.cat(Y).numpy()


def separability(feature_sets, seed=6304):
    """feature_sets: list of arrays, one per domain. Each is subsampled to the size of the
    smallest so the probe is balanced, then a stratified 70/30 split is used."""
    rng = np.random.RandomState(seed)
    n = min(len(f) for f in feature_sets)
    X = np.concatenate([f[rng.choice(len(f), n, replace=False)] for f in feature_sets])
    d = np.repeat(np.arange(len(feature_sets)), n)
    Xtr, Xte, dtr, dte = train_test_split(X, d, test_size=0.3, stratify=d, random_state=seed)
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    probe.fit(Xtr, dtr)
    return float((probe.predict(Xte) == dte).mean() * 100.0)
