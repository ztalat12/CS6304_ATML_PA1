"""The freeze record: proof that nothing was tuned on the unknowns.

Handout: "Unknown examples may be evaluated only after every checkpoint and score definition has been fixed"
and "no real unknown examples influenced Task 4 training or threshold selection".

`task4.freeze` writes results/frozen/frozen.json once all three models are trained. It stores the SHA-256 of
  * every selected checkpoint (best.pt),
  * every file that defines a score, a threshold, the features/logits themselves (network, eval transform,
    extraction) or the unknown grouping (FROZEN_CODE below),
  * the Mahalanobis statistics (class means + shared diagonal variance),
  * the thresholds themselves (the numbers are in the record).
`verify_frozen` recomputes those hashes and stops the program if anything differs. CIFAR-100 can only be
loaded through `load_unknowns`, which calls `verify_frozen` first, so an unknown image can never be seen while
something it could influence is still open to change. Every later access is logged with a timestamp.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

from common.io import load_json, resolve, save_json

# Files whose contents DEFINE the scores, thresholds and unknown grouping. Editing any of them after the
# freeze makes verify_frozen fail - that is the point.
FROZEN_CODE = [
    # the scores and the threshold rule
    "task4/scores/__init__.py", "task4/scores/msp.py", "task4/scores/mls.py", "task4/scores/energy.py",
    "task4/scores/mahalanobis.py", "task4/scores/proser_detection.py",
    "task4/evaluation/thresholds.py", "task4/configs/proser.yaml",
    # what f(x) and z(x) ARE for any input: the network definition, the eval transform / normalisation,
    # and the extraction code
    "task4/models/resnet_cifar.py", "task4/data/cifar10.py", "task4/extract_outputs.py",
    # the fixed unknown grouping
    "task4/data/cifar100_unknowns.py",
]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(resolve(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path) -> str:
    """Hash of a source file with line endings normalised (so a Windows CRLF checkout verifies too)."""
    with open(resolve(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def sha256_arrays(npz_path) -> str:
    """Hash of the ARRAYS in an .npz (names, dtypes, shapes, values) - not of the zip container, whose bytes
    include a timestamp and would change every time the same numbers are saved again."""
    import numpy as np
    h = hashlib.sha256()
    with np.load(resolve(npz_path)) as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def rel(path) -> str:
    """Store paths relative to the repository root when possible (the record then verifies on any machine)."""
    import os
    from common.io import repo_root
    p = os.path.abspath(resolve(path))
    r = os.path.relpath(p, repo_root())
    return p if r.startswith("..") else r


def code_hashes() -> dict:
    return {p: sha256_text(p) for p in FROZEN_CODE}


def verify_frozen(frozen_file, quiet=False) -> dict:
    """Recompute every hash in the freeze record. Returns the record if all match, otherwise stops."""
    fp = Path(resolve(frozen_file))
    if not fp.exists():
        raise SystemExit(f"LEAKAGE GUARD: {fp} does not exist. Train all models and run `python -m task4.freeze` "
                         "BEFORE anything touches CIFAR-100.")
    rec = load_json(fp)
    problems = []
    for name, ck in rec["checkpoints"].items():
        if sha256_file(ck["path"]) != ck["sha256"]:
            problems.append(f"checkpoint {name} ({ck['path']}) changed after the freeze")
    for p, h in rec["code_sha256"].items():
        if sha256_text(p) != h:
            problems.append(f"{p} changed after the freeze")
    for name, st in rec.get("mahalanobis_stats", {}).items():
        if sha256_arrays(st["path"]) != st["sha256"]:
            problems.append(f"Mahalanobis statistics of {name} changed after the freeze")
    if problems:
        raise SystemExit("LEAKAGE GUARD: the frozen state was modified:\n  " + "\n  ".join(problems))
    if not quiet:
        print(f"freeze record verified: {len(rec['checkpoints'])} checkpoints, {len(rec['code_sha256'])} code files, "
              f"{len(rec.get('mahalanobis_stats', {}))} Mahalanobis statistics files unchanged since {rec['frozen_at']}")
    return rec


def log_unknown_access(frozen_file, n_images: int) -> None:
    """Append one line per CIFAR-100 load to results/frozen/unknown_access_log.json."""
    rec = load_json(resolve(frozen_file))
    log_file = Path(resolve(frozen_file)).with_name("unknown_access_log.json")
    log = load_json(log_file) if log_file.exists() else {"frozen_at": rec["frozen_at"], "accesses": []}
    log["accesses"].append({"time": now(), "n_images": n_images})
    save_json(log, log_file)
    first = log["accesses"][0]["time"]
    assert first >= rec["frozen_at"], "first CIFAR-100 access predates the freeze record"
    print(f"CIFAR-100 unknowns loaded ({n_images} images). Access #{len(log['accesses'])}; "
          f"freeze {rec['frozen_at']} <= first access {first}")


def pretty(rec: dict) -> str:
    return json.dumps({k: v for k, v in rec.items() if k not in ("code_sha256",)}, indent=2)
