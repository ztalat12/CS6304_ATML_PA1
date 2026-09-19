"""Common training loop for Tasks 2 and 3.

Every method (Source-only/ERM, DAN, DANN, CDAN, DAN-DG, SAM) runs through THIS loop, so
initialisation, sampling, augmentation, optimiser, budget, BN policy and checkpoint rule
are identical by construction. A method only defines `training_step`.

Batch composition per update:  8 x photo + 8 x art_painting + 8 x cartoon (= 24 source)
                               + 24 target images (Task 2 adaptation methods only).
One "epoch" = floor(total source-train images / 24) updates (one pass worth of source data).
Checkpoint selection: best MEAN macro-F1 over the three source validation splits;
early stopping after `patience` epochs without improvement. No target labels anywhere.
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.io import save_json
from common.metrics import classification_metrics
from common.seed import make_generator, seed_worker
from shared.models import set_train_mode
from shared.pacs import CLASSES, SOURCE_DOMAINS, PACSImages, eval_transform, train_transform


class Method(nn.Module):
    """Base class. Subclasses implement `loss(...)` or override `training_step(...)`.
    Own sub-modules (e.g. a domain discriminator) are registered on the method and are
    optimised together with the network."""
    uses_target = False
    grad_clip = 0.0            # set by fit() from cfg["grad_clip"]; 0 disables clipping
    _clip_params = None        # set by fit(): network parameters + the method's own parameters

    def loss(self, model, xs, ys, ds, xt, progress):
        raise NotImplementedError

    def clip_(self) -> float:
        """GRADIENT CLIPPING (applied identically to EVERY method, so the comparison stays fair).

        Why it is needed: the gradient-reversal layer makes the backbone *maximise* the domain
        cross-entropy, which is unbounded above. The backbone can raise it simply by scaling its
        features up; the discriminator then chases with larger weights, which scales the features'
        gradients again -- a runaway that also blows up the classification loss (this is exactly
        what happened without clipping: loss_cls ~ 50 after one epoch, then 1e5). Rescaling the
        whole gradient to a maximum norm bounds the step size without changing the objective.
        Returns the PRE-clipping total gradient norm, which is logged so you can show in the
        report how often clipping was active."""
        if not self._clip_params:
            return float("nan")
        total = torch.nn.utils.clip_grad_norm_(self._clip_params, self.grad_clip or float("inf"))
        return float(total)

    def training_step(self, model, optimizer, xs, ys, ds, xt, progress):
        loss, logs = self.loss(model, xs, ys, ds, xt, progress)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        logs["grad_norm"] = self.clip_()
        optimizer.step()
        logs["loss_total"] = loss.item()
        return logs


class Forever:
    """Cycle a DataLoader indefinitely (re-shuffles each time it wraps)."""

    def __init__(self, loader):
        self.loader, self.it = loader, iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def _loader(ds, bs, seed, shuffle, num_workers, drop_last):
    # A private generator per loader -> sampling AND (with num_workers>0) augmentation
    # randomness are isolated from the global RNG and from other loaders.
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
                      generator=make_generator(seed), worker_init_fn=seed_worker,
                      persistent_workers=num_workers > 0, pin_memory=True)


def build_loaders(cfg, splits, target_items=None):
    root, seed, nw = cfg["data_root"], cfg["seed"], cfg["num_workers"]
    assert nw >= 1, "use num_workers>=1 so augmentation RNG is isolated from dropout/global RNG"
    src = {d: Forever(_loader(PACSImages(root, splits[d]["train"], train_transform()),
                              cfg["per_domain_batch"], seed + i, True, nw, True))
           for i, d in enumerate(SOURCE_DOMAINS)}
    val = {d: DataLoader(PACSImages(root, splits[d]["val"], eval_transform()), batch_size=128,
                         shuffle=False, num_workers=nw) for d in SOURCE_DOMAINS}
    tgt = None
    if target_items is not None:   # unlabeled: labels never returned
        tgt = Forever(_loader(PACSImages(root, target_items, train_transform(), return_labels=False),
                              cfg["target_batch"], seed + 100, True, nw, True))
    steps = sum(len(splits[d]["train"]) for d in SOURCE_DOMAINS) // (3 * cfg["per_domain_batch"])
    return src, val, tgt, steps


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def evaluate_sources(model, val_loaders, device):
    out = {}
    for d, loader in val_loaders.items():
        y, p = evaluate_loader(model, loader, device)
        m = classification_metrics(y, p, len(CLASSES))
        out[d] = {"acc": m["acc"], "macro_f1": m["macro_f1"]}
    accs = [out[d]["acc"] for d in val_loaders]
    f1s = [out[d]["macro_f1"] for d in val_loaders]
    out["mean"] = {"acc": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}
    out["worst"] = {"acc": float(np.min(accs)), "macro_f1": float(np.min(f1s))}
    return out


def fit(method: Method, model, cfg, splits, out_dir, device, target_items=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src, val, tgt, steps_per_epoch = build_loaders(cfg, splits, target_items if method.uses_target else None)
    total_steps = cfg["max_epochs"] * steps_per_epoch      # DANN/CDAN schedule progress p is w.r.t. the full budget
    method.to(device)
    params = list(model.parameters()) + list(method.parameters())
    # AdamW is the handout's optimiser. `optimizer: sgd` exists only for the DANN stability
    # diagnostic: Ganin et al. used SGD with momentum and an annealed learning rate, and unlike Adam,
    # SGD's step size shrinks with the gradient, which is what alpha and clipping implicitly assume.
    if str(cfg.get("optimizer", "adamw")).lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=float(cfg.get("sgd_momentum", 0.9)),
                                    weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    base_lr, lr_sched = float(cfg["lr"]), str(cfg.get("lr_schedule", "none")).lower()
    method._clip_params, method.grad_clip = params, float(cfg.get("grad_clip", 0.0))

    best, wait, step, history = -1.0, 0, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        set_train_mode(model)                  # train mode + frozen BN running statistics
        method.train()
        logs, t0 = {}, time.time()
        for _ in range(steps_per_epoch):
            xs, ys, ds = [], [], []
            for i, d in enumerate(SOURCE_DOMAINS):             # domain-balanced source batch
                x, y = src[d].next()
                xs.append(x); ys.append(y); ds.append(torch.full_like(y, i))
            xs, ys, ds = torch.cat(xs).to(device), torch.cat(ys).to(device), torch.cat(ds).to(device)
            xt = tgt.next().to(device) if tgt is not None else None
            p_now = step / total_steps
            if lr_sched == "ganin":                        # mu_p = mu_0 / (1 + 10 p)^0.75
                for grp in optimizer.param_groups:
                    grp["lr"] = base_lr / (1.0 + 10.0 * p_now) ** 0.75
            out = method.training_step(model, optimizer, xs, ys, ds, xt, progress=p_now)
            for k, v in out.items():
                logs.setdefault(k, []).append(v)
            step += 1
        val_metrics = evaluate_sources(model, val, device)
        record = {"epoch": epoch, "step": step, "time_s": time.time() - t0,
                  **{k: float(np.mean(v)) for k, v in logs.items()},
                  **{f"val_{d}_{k}": v for d, m in val_metrics.items() for k, v in m.items()}}
        history.append(record)
        save_json(history, out_dir / "history.json")      # saved every epoch, BEFORE a possible early stop
        score = val_metrics["mean"]["macro_f1"]
        print(f"[{cfg['run_name']}] ep{epoch:02d} " + " ".join(
            f"{k}={v:.4f}" for k, v in record.items() if k.startswith(("loss", "disc", "alpha"))) +
              f" | val mean F1={score:.2f}")
        if score > best:
            best, wait = score, 0
            torch.save({"model": copy.deepcopy(model.state_dict()), "method": method.state_dict(),
                        "epoch": epoch, "val": val_metrics, "config": cfg}, out_dir / "best.pt")
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"early stop at epoch {epoch} (best mean macro-F1 {best:.2f})")
                break
    save_json({"best_mean_macro_f1": best, "epochs_run": len(history), "steps_per_epoch": steps_per_epoch,
               "config": cfg}, out_dir / "summary.json")
    return history
