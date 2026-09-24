"""Save each frozen model's penultimate features f(x) and logits z(x) (+ PROSER dummy logits) to the cache.

  # stage 1 - CIFAR-10 only (train / val / test); run after all three models are trained
  python -m task4.extract_outputs --stage known   --config task4/configs/base.yaml --set data_root=...
  # stage 2 - CIFAR-100 unknowns; REFUSES to run until task4.freeze has written the freeze record
  python -m task4.extract_outputs --stage unknown --config task4/configs/base.yaml --set data_root=...
  (add --skip-existing to keep files already extracted from the same checkpoints, e.g. after a crash)

Handout (Vanilla step): "Freeze the selected checkpoint and extract its penultimate feature f(x) and logits
z(x) for the CIFAR-10 training, validation, and test examples and the fixed CIFAR-100 evaluation examples."
Every score is later computed from these SAME saved arrays, so all scores see identical numbers.

Details
  * The eval pipeline only (ToTensor + Normalize): no augmentation, including for the training features that
    define the Mahalanobis statistics ("unaugmented CIFAR-10 training features").
  * model.eval() and float32 (no mixed precision), same batch size and memory layout as the validation pass in
    train.py - so the validation accuracy recomputed from the saved logits must equal the checkpoint's recorded
    validation accuracy. That check is printed; a mismatch stops the program.
  * cache/<run>/<split>.npz holds: features (N x 512, float32), logits (N x 10), dummy_logits (N x 5, PROSER
    only), labels (CIFAR-10 class, or group code 1 = near / 2 = far for unknowns), index (position in the
    source dataset). cache/<run>/manifest.json records the SHA-256 of the checkpoint the arrays came from.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.io import apply_overrides, load_config, load_json, resolve, save_json
from common.seed import get_device, set_seed
from task4.audit import sha256_file
from task4.data.cifar10 import build_transform, known_split
from task4.models.resnet_cifar import load_model

RUNS = ["vanilla", "gcsc", "proser"]
FROZEN_FILE = "task4/results/frozen/frozen.json"


@torch.no_grad()
def extract(model, ds, device, batch_size, num_workers, channels_last):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    F, Z, D, Y = [], [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        f, z, d = model.forward_all(x)
        F.append(f.float().cpu()); Z.append(z.float().cpu()); Y.append(torch.as_tensor(y))
        if d is not None:
            D.append(d.float().cpu())
    out = {"features": torch.cat(F).numpy(), "logits": torch.cat(Z).numpy(), "labels": torch.cat(Y).numpy()}
    if D:
        out["dummy_logits"] = torch.cat(D).numpy()
    return out


def load_cached(cache_dir, run, split, expect_sha=None):
    """Load one cache file; if expect_sha is given, check that it came from that exact checkpoint."""
    d = Path(resolve(cache_dir)) / run
    if expect_sha is not None:
        man = load_json(d / "manifest.json")
        if man["checkpoint_sha256"] != expect_sha:
            raise SystemExit(f"cache/{run} was extracted from a different checkpoint than the frozen one - "
                             f"re-run `extract_outputs --stage known`.")
    z = np.load(d / f"{split}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def cached_from(cache, run, sha, files):
    """True if cache/<run> already holds `files` extracted from the checkpoint with this SHA-256."""
    man = cache / run / "manifest.json"
    return man.exists() and load_json(man)["checkpoint_sha256"] == sha and all((cache / run / f).exists() for f in files)


def main(cfg, stage, runs, skip_existing=False):
    cfg["data_root"] = resolve(cfg["data_root"])
    device = get_device()
    set_seed(cfg["seed"])
    channels_last = bool(cfg.get("channels_last")) and device.type == "cuda"
    runs_dir, cache = Path(resolve(cfg["results_dir"])), Path(resolve(cfg["cache_dir"]))
    dbg = int(cfg.get("debug_subset") or 0)
    tf = build_transform("eval")

    if stage == "unknown":
        from task4.data.cifar100_unknowns import load_unknowns          # imported ONLY in this stage
        unknowns = load_unknowns(cfg["data_root"], tf, cfg.get("frozen_file", FROZEN_FILE))  # verifies the freeze
        frozen = load_json(resolve(cfg.get("frozen_file", FROZEN_FILE)))

    for run in runs:
        ckpt = runs_dir / run / "best.pt"
        model, ck = load_model(ckpt, device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        sha = sha256_file(ckpt)
        print(f"--- {run}: {ckpt} (epoch {ck['epoch']}, recorded val acc {ck['val_acc']:.2f}%, "
              f"dummies {ck.get('n_dummies', 0)}, sha256 {sha[:12]}...)")
        (cache / run).mkdir(parents=True, exist_ok=True)
        needed = ["train.npz", "val.npz", "test.npz"] if stage == "known" else ["unknown.npz"]
        if skip_existing and cached_from(cache, run, sha, needed):
            print(f"  {', '.join(needed)} already extracted from this exact checkpoint - kept as they are")
            continue

        if stage == "known":
            for split in ["train", "val", "test"]:
                t0 = time.time()
                ds = known_split(cfg["data_root"], resolve(cfg["split_file"]), split, tf, dbg)
                out = extract(model, ds, device, cfg["eval_batch_size"], cfg["num_workers"], channels_last)
                out["index"] = np.asarray(ds.indices)
                np.savez(cache / run / f"{split}.npz", **out)
                acc = 100.0 * np.mean(out["logits"].argmax(1) == out["labels"])
                print(f"  {split:5s}: {len(out['labels']):6d} images | features {out['features'].shape} | "
                      f"accuracy of the 10 known logits {acc:6.2f}% | {time.time() - t0:.0f}s")
                if split == "val":
                    n_off = round(abs(acc - ck["val_acc"]) * len(out["labels"]) / 100)
                    if n_off > 2:
                        raise SystemExit(f"validation accuracy from the saved logits ({acc:.4f}) != the checkpoint's "
                                         f"recorded value ({ck['val_acc']:.4f}) - the extraction is not faithful.")
                    val_check = "identical" if n_off == 0 else f"differs by {n_off} image(s) (floating-point ties)"
            save_json({"run": run, "checkpoint": str(ckpt), "checkpoint_sha256": sha, "epoch": ck["epoch"],
                       "val_acc": ck["val_acc"], "debug_subset": dbg}, cache / run / "manifest.json")
            print(f"  check: val accuracy from the saved logits vs the checkpoint's recorded val accuracy: {val_check}")
        else:
            if frozen["checkpoints"][run]["sha256"] != sha:
                raise SystemExit(f"{run}: checkpoint differs from the frozen one")
            t0 = time.time()
            out = extract(model, unknowns, device, cfg["eval_batch_size"], cfg["num_workers"], channels_last)
            out["index"], out["fine_class"], out["group"] = unknowns.index, unknowns.fine_class, unknowns.group
            np.savez(cache / run / "unknown.npz", **{k: (v.astype("U") if v.dtype.kind in "OU" else v)
                                                      for k, v in out.items()})
            print(f"  unknown: {len(out['labels'])} images (near {int((out['group'] == 'near').sum())}, far "
                  f"{int((out['group'] == 'far').sum())}) | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["known", "unknown"], required=True)
    ap.add_argument("--config", default="task4/configs/base.yaml")
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep cache files already extracted from the same checkpoint (crash recovery)")
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set), a.stage, a.runs, a.skip_existing)
