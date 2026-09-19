"""Reproducibility helpers shared by every task.

WHY THIS MATTERS: the assignment asks for *controlled* comparisons. If two methods
differ in their random initialisation, sampling order or augmentation, you cannot
attribute a difference in results to the method itself. Fixing seeds (6304 everywhere)
removes most of that noise. GPU kernels can still be slightly non-deterministic, so
mention this in the report as a residual source of variation.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 6304, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN autotuning picks different kernels run-to-run -> disable it.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """A *private* RNG stream. Giving each DataLoader its own generator means that
    one loader's consumption of random numbers cannot shift another loader's order
    (e.g. drawing target batches in DANN does not change the source batch order)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn. PyTorch already seeds torch inside each worker from
    the loader's generator; we propagate that seed to NumPy/random so that any
    augmentation using them is reproducible as well."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
