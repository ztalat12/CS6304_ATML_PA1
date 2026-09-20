"""Train one Task 3 configuration (DAN-DG or SAM). ERM is NOT retrained.

  python -m task3.train --config task3/configs/dan_dg.yaml
  python -m task3.train --config task3/configs/sam.yaml
  # controlled study (choose one family):
  python -m task3.train --config task3/configs/study_sam_rho0.01.yaml
  python -m task3.train --config task3/configs/study_sam_rho0.1.yaml

This file deliberately never imports `load_target_items`: no Sketch path can reach
training, diagnostics or checkpoint selection.
"""
import argparse
from pathlib import Path

from common.io import apply_overrides, load_config, load_json, resolve, save_json
from common.seed import get_device, set_seed
from shared.engine import fit
from shared.models import ResNet18Classifier
from shared.pacs_protocol import load_source_splits
from task3.methods.dan_dg import DANDG
from task3.methods.sam import SAM

METHODS = {"dan_dg": DANDG, "sam": SAM}

# The handout requires Task 3 to reuse Task 2's protocol exactly, because its ERM baseline IS the
# Task 2 Source-only checkpoint. Comparing SAM/DAN-DG against an ERM trained under different
# settings would confound the method with the pipeline, so these keys must match.
PROTOCOL_KEYS = ["seed", "num_workers", "per_domain_batch", "lr", "weight_decay", "grad_clip",
                 "max_epochs", "patience"]


def check_protocol(cfg):
    ref = Path(resolve(cfg["erm_checkpoint"])).parent / "config_resolved.json"
    if not ref.exists():
        print(f"WARNING: {ref} not found -- cannot verify that Task 3 matches the ERM protocol.")
        return
    erm = load_json(ref)
    keys = PROTOCOL_KEYS
    if cfg.get("seed_check"):
        # Robustness run only (e.g. --set seed=6305 run_name=dan_dg_seed6305 seed_check=true):
        # the seed differs from ERM's ON PURPOSE to test whether a result is luck of seed 6304.
        # Such a run is never part of a comparison; every other protocol key is still checked.
        keys = [k for k in PROTOCOL_KEYS if k != "seed"]
        print(f"SEED CHECK: seed {cfg.get('seed')} differs from the ERM checkpoint's seed {erm.get('seed')} on "
              "purpose. Robustness run -- not part of any comparison.")
    bad = {k: (erm.get(k), cfg.get(k)) for k in keys if erm.get(k) != cfg.get(k)}
    if bad:
        raise SystemExit("Task 3 protocol differs from the ERM checkpoint it will be compared against:\n"
                         + "\n".join(f"  {k}: ERM={a!r}  Task3={b!r}" for k, (a, b) in bad.items())
                         + "\nFix task2/configs/base.yaml (or retrain Source-only) so both match.")
    print("Protocol matches the Task 2 Source-only checkpoint:",
          ", ".join(f"{k}={cfg[k]}" for k in keys))


def main(cfg):
    if cfg["method"]["name"] == "erm":
        raise SystemExit("ERM is the Task 2 Source-only checkpoint -- reuse it, do not retrain.")
    cfg["data_root"] = resolve(cfg["data_root"])
    check_protocol(cfg)
    device = get_device()
    set_seed(cfg["seed"])
    model = ResNet18Classifier(num_classes=7).to(device)
    mcfg = dict(cfg["method"])
    method = METHODS[mcfg.pop("name")](**mcfg)
    assert not method.uses_target
    out_dir = Path(resolve(cfg["results_dir"])) / cfg["run_name"]
    save_json(cfg, out_dir / "config_resolved.json")
    fit(method, model, cfg, load_source_splits(), out_dir, device, target_items=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set))
