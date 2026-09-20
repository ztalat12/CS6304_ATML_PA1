"""ERM baseline for Task 3.

L_ERM = 1/3 * sum_{e in {P, A, C}} R_e(theta). With domain-balanced batches (8/8/8) the
plain mean cross-entropy over the 24 images IS this objective, i.e. it is exactly Task 2's
Source-only model. The handout requires REUSING that checkpoint, so there is nothing to
train here -- this module only resolves the checkpoint path.
"""
from pathlib import Path

from common.io import resolve


def erm_checkpoint(cfg) -> Path:
    p = Path(resolve(cfg["erm_checkpoint"]))
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m task2.train --config task2/configs/source_only.yaml` first")
    return p
