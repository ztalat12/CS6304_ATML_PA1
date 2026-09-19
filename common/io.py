"""Small I/O helpers: JSON that understands NumPy/Torch, and YAML configs with inheritance.

Config inheritance: a YAML file may contain `base: other.yaml`. The base file is loaded
first and the child's keys override it (recursively for nested dicts). This keeps every
method config tiny and guarantees that shared settings (optimizer, epochs, seed, ...)
are literally identical across methods -- one of the assignment's fairness requirements.
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_serialisable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "item") and callable(obj.item):  # 0-d torch tensor
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_json(obj, path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(_to_serialisable(obj), f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_cfg = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_update(base_cfg, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides) -> dict:
    """CLI overrides like `method.lambda_mmd=10` (values parsed as YAML scalars)."""
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path) -> str:
    """Resolve config paths relative to the repository root."""
    p = Path(os.path.expanduser(str(path)))
    return str(p if p.is_absolute() else repo_root() / p)
