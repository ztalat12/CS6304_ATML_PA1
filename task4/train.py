"""Train one Task 4 model (Vanilla, GCSC or PROSER) on the 45k CIFAR-10 training split.

  python -m task4.train --config task4/configs/vanilla.yaml --set data_root=/path/to/cifar
  python -m task4.train --config task4/configs/gcsc.yaml    --set data_root=/path/to/cifar
  python -m task4.train --config task4/configs/proser.yaml  --set data_root=/path/to/cifar   # needs vanilla/best.pt

One shared loop for all three models, so the only differences between runs are the ones in their configs:
  * SGD (momentum 0.9, weight decay 5e-4), cosine decay over the run, batch size 128, seed 6304;
  * after every epoch, accuracy on the 5,000-image CIFAR-10 VALIDATION split;
  * the checkpoint with the highest validation accuracy is saved as best.pt (ties keep the earlier epoch).

Nothing here can see CIFAR-100: this file never imports task4.data.cifar100_unknowns, and the CIFAR-10 test set
is not used either (it is only read later, by extract_outputs.py, after training has finished).

Outputs in <results_dir>/<run_name>/:
  config_resolved.json   the exact settings used
  history.json           one row per epoch (losses, train/val accuracy, learning rate, time)
  best.pt                the selected checkpoint (git-ignored)
  last.pt                the final epoch's weights (git-ignored; used only by the post-hoc BatchNorm diagnosis)
  summary.json           best epoch, best validation accuracy, run time, hardware, finished flag
"""
import argparse
import math
import platform
import time
from pathlib import Path

import torch

from common.io import apply_overrides, load_config, resolve, save_json
from common.seed import get_device, set_seed
from task4.data.cifar10 import build_transform, known_split, labels_of, make_loader
from task4.methods import METHODS
from task4.methods.bn_recalibration import recalibrate_bn


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):                 # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(model, loader, device, channels_last):
    """Validation accuracy on the ten KNOWN logits (float32, no augmentation). For PROSER also the share of
    known validation images whose strongest dummy beats every known logit ('dummy wins') - a warning sign if
    it grows, because the dummy would then be rejecting real CIFAR-10 images."""
    model.eval()
    correct = dummy_wins = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        z, d = model.heads(model.features(x))
        correct += (z.argmax(1) == y).sum().item()
        if d is not None:
            dummy_wins += (d.max(1).values > z.max(1).values).sum().item()
        n += len(y)
    out = {"val_acc": 100.0 * correct / n}
    if model.dummy is not None:
        out["val_dummy_wins"] = 100.0 * dummy_wins / n
    return out


def main(cfg, force=False):
    cfg["data_root"] = resolve(cfg["data_root"])
    out_dir = Path(resolve(cfg["results_dir"])) / cfg["run_name"]
    if (out_dir / "summary.json").exists() and (out_dir / "best.pt").exists() and not force:
        print(f"{out_dir} already finished - skipping (pass --force to retrain).")
        return
    device = get_device()
    set_seed(cfg["seed"])                                   # also turns on deterministic cuDNN

    mcfg = dict(cfg["method"])
    method = METHODS[mcfg.pop("name")](**mcfg, seed=cfg["seed"])
    model = method.build_model(cfg).to(device)
    channels_last = bool(cfg.get("channels_last")) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    amp = bool(cfg.get("amp")) and device.type == "cuda"

    dbg = int(cfg.get("debug_subset") or 0)
    train_ds = known_split(cfg["data_root"], resolve(cfg["split_file"]), "train", method.train_transform(), dbg)
    val_ds = known_split(cfg["data_root"], resolve(cfg["split_file"]), "val", build_transform("eval"), dbg)
    train_loader = make_loader(train_ds, cfg["batch_size"], True, cfg["num_workers"], cfg["seed"])
    val_loader = make_loader(val_ds, cfg["eval_batch_size"], False, cfg["num_workers"], cfg["seed"])
    # Opt-in (post-hoc PROSER variant only): re-estimate BatchNorm statistics from real training images after every
    # epoch, before validation and checkpointing. Off unless the config sets bn_recalibrate_batches > 0.
    recal = int(cfg.get("bn_recalibrate_batches") or 0)
    if recal:
        recal_ds = known_split(cfg["data_root"], resolve(cfg["split_file"]), "train", build_transform("train"), dbg)
        recal_loader = make_loader(recal_ds, cfg["batch_size"], True, cfg["num_workers"], cfg["seed"] + 1)

    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"])
    assert cfg["scheduler"] == "cosine"
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = grad_scaler(amp)

    save_json(cfg, out_dir / "config_resolved.json")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== {cfg['run_name']} ===")
    print(f"  method        : {method.describe()}")
    print(f"  device        : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
          f" | AMP {amp} | channels_last {channels_last} | cuDNN deterministic {torch.backends.cudnn.deterministic}")
    print(f"  data          : train {len(train_ds)} images (classes {sorted(set(labels_of(train_ds).tolist()))}),"
          f" val {len(val_ds)} images" + (f"   [DEBUG SUBSET {dbg}]" if dbg else ""))
    print(f"  optimisation  : SGD lr {cfg['lr']} momentum {cfg['momentum']} wd {cfg['weight_decay']}, cosine over "
          f"{cfg['epochs']} epochs, batch {cfg['batch_size']}, {len(train_loader)} steps/epoch, seed {cfg['seed']}")
    print(f"  parameters    : {n_params / 1e6:.2f} M")
    print(f"  checkpoint    : highest CIFAR-10 validation accuracy -> {out_dir / 'best.pt'}\n", flush=True)

    history, best, best_epoch, t_start = [], -1.0, 0, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        lr = opt.param_groups[0]["lr"]
        sums, n_steps, correct, seen, nonfinite = {}, 0, 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with autocast(device, amp):
                loss, logs, z_acc, y_acc = method.training_step(model, x, y)
            if not torch.isfinite(loss):
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            logs["loss"] = loss.detach()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v.detach().float().to(device)
            correct += (z_acc.argmax(1) == y_acc).sum()
            seen += len(y_acc)
            n_steps += 1
        sched.step()
        if recal:
            recalibrate_bn(model, recal_loader, device, recal, channels_last)
        if nonfinite > 0.1 * len(train_loader):
            raise RuntimeError(f"{nonfinite} of {len(train_loader)} steps had a non-finite loss - training diverged.")

        row = {"epoch": epoch, "lr": lr, **{k: float(v) / max(n_steps, 1) for k, v in sums.items()},
               "train_acc": 100.0 * float(correct) / max(seen, 1), "nonfinite_steps": nonfinite}
        row.update(evaluate(model, val_loader, device, channels_last))
        row["epoch_sec"] = time.time() - t0
        improved = row["val_acc"] > best
        if improved:
            best, best_epoch = row["val_acc"], epoch
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": state, "epoch": epoch, "val_acc": best, "method": method.name,
                        "n_dummies": getattr(model, "n_dummies", 0), "num_classes": model.num_classes,
                        "config": cfg}, out_dir / "best.pt")
        row["best_val_acc"], row["best_epoch"] = best, best_epoch
        history.append(row)
        save_json(history, out_dir / "history.json")

        extra = ""
        if "cp_ce" in row:
            extra = (f" | CE {row['cp_ce']:.3f} dummyCE {row['cp_dummy']:.3f} mixCE {row['dp']:.3f}"
                     f" pairs {row['pairs']:.1f} | val dummy-wins {row['val_dummy_wins']:.2f}%")
        eta = (time.time() - t_start) / epoch * (cfg["epochs"] - epoch) / 60
        print(f"ep {epoch:3d}/{cfg['epochs']} | lr {lr:.5f} | loss {row['loss']:.4f} | train {row['train_acc']:6.2f}%"
              f" | val {row['val_acc']:6.2f}%{' *' if improved else '  '} (best {best:.2f} @ {best_epoch})"
              f"{extra} | {row['epoch_sec']:.0f}s | ETA {eta:.0f} min"
              + (f" | {nonfinite} non-finite steps skipped" if nonfinite else ""), flush=True)

    # the final epoch's weights too (the BatchNorm diagnosis needs them); best.pt is unaffected
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": state, "epoch": cfg["epochs"], "val_acc": history[-1]["val_acc"], "method": method.name,
                "n_dummies": getattr(model, "n_dummies", 0), "num_classes": model.num_classes,
                "config": cfg}, out_dir / "last.pt")
    minutes = (time.time() - t_start) / 60
    save_json({"run_name": cfg["run_name"], "method": method.name, "finished": True, "best_epoch": best_epoch,
               "best_val_acc": best, "epochs": cfg["epochs"], "train_minutes": minutes,
               "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU",
               "torch": torch.__version__, "amp": amp, "debug_subset": dbg,
               "last_val_acc": history[-1]["val_acc"]}, out_dir / "summary.json")
    print(f"\nfinished {cfg['run_name']} in {minutes:.1f} min: best val acc {best:.2f}% at epoch {best_epoch} "
          f"(last epoch {history[-1]['val_acc']:.2f}%)")
    if not math.isfinite(best) or best < 0:
        raise RuntimeError("no checkpoint was saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set), force=a.force)
