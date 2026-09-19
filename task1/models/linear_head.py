"""Linear probe ("classifier head") trained on cached frozen features.

Because the backbone is frozen, features are computed once and the head is trained on
the cached tensors -- fast, and guarantees the head is the only thing that learns.
Recipe fixed by the handout: AdamW(lr=1e-3, wd=1e-4), <= 50 epochs, early stopping after
5 epochs without a better validation accuracy, seed 6304. No augmentation (the features
of the un-augmented canvases are used).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.seed import make_generator, set_seed


@torch.no_grad()
def _accuracy(head, X, y, device, bs=4096):
    head.eval()
    correct = 0
    for s in range(0, len(X), bs):
        correct += (head(X[s:s + bs].to(device)).argmax(1).cpu() == y[s:s + bs]).sum().item()
    return 100.0 * correct / len(X)


def train_linear_head(Xtr, ytr, Xva, yva, num_classes, cfg, seed, device):
    set_seed(seed)                                   # identical init/order for every backbone
    head = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = make_generator(seed)
    best_acc, best_state, best_epoch, wait, history = -1.0, None, 0, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        head.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        total = 0.0
        for s in range(0, len(perm), cfg["batch_size"]):
            b = perm[s:s + cfg["batch_size"]]
            loss = F.cross_entropy(head(Xtr[b].to(device)), ytr[b].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        va = _accuracy(head, Xva, yva, device)
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_acc": va})
        if va > best_acc:                            # strict improvement resets patience
            best_acc, best_state, best_epoch, wait = va, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_epoch": best_epoch, "best_val_acc": best_acc, "history": history}
