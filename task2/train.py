"""Train one Task 2 configuration.

  python -m task2.train --config task2/configs/source_only.yaml
  python -m task2.train --config task2/configs/dan.yaml
  python -m task2.train --config task2/configs/dann.yaml
  python -m task2.train --config task2/configs/cdan.yaml
  # controlled study (choose one family):
  python -m task2.train --config task2/configs/study_dan_lambda0.1.yaml
  python -m task2.train --config task2/configs/study_dan_lambda10.yaml

Transductive UDA: the WHOLE Sketch domain is the unlabeled adaptation set. Its labels are
stripped inside the dataset (return_labels=False) and are only used by evaluate_final.py.
"""
import argparse
from pathlib import Path

from common.io import apply_overrides, load_config, resolve, save_json
from common.seed import get_device, set_seed
from shared.engine import fit
from shared.models import ResNet18Classifier
from shared.pacs_protocol import load_source_splits, load_target_items
from task2.methods.cdan import CDAN
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.source_only import SourceOnly

METHODS = {"source_only": SourceOnly, "dan": DAN, "dann": DANN, "cdan": CDAN}


def main(cfg):
    cfg["data_root"] = resolve(cfg["data_root"])
    device = get_device()
    set_seed(cfg["seed"])
    model = ResNet18Classifier(num_classes=7).to(device)   # identical head init for every method
    mcfg = dict(cfg["method"])
    method = METHODS[mcfg.pop("name")](**mcfg)             # created AFTER the model
    out_dir = Path(resolve(cfg["results_dir"])) / cfg["run_name"]
    save_json(cfg, out_dir / "config_resolved.json")
    target = load_target_items() if method.uses_target else None
    fit(method, model, cfg, load_source_splits(), out_dir, device, target_items=target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. max_epochs=2")
    a = ap.parse_args()
    main(apply_overrides(load_config(a.config), a.set))
