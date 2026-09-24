"""CIFAR-10: the ten KNOWN classes of Task 4.

Three image pipelines, all ending in the same normalisation:

  train  RandomCrop(32, padding=4) -> RandomHorizontalFlip -> ToTensor -> Normalize      (Vanilla, PROSER)
  gcsc   RandomCrop(32, padding=4) -> RandomHorizontalFlip -> RandAugment(2, 9)
                                                          -> ToTensor -> Normalize      (GCSC)
  eval   ToTensor -> Normalize                        (validation, test, feature extraction, unknowns)

The handout says RandAugment goes "after the crop and flip and before conversion and normalization", which
is exactly where `gcsc` puts it: RandAugment works on the PIL image, before it becomes a tensor.

Normalisation uses the CIFAR-10 training-set channel statistics. The CIFAR-100 unknowns are pushed through
the SAME `eval` pipeline (same mean/std), because at test time the model cannot know which dataset an image
came from - it must treat every input the same way.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common.io import load_json
from common.seed import make_generator, seed_worker

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = (0.4914, 0.4822, 0.4465)   # CIFAR-10 training-set channel means
STD = (0.2470, 0.2435, 0.2616)    # CIFAR-10 training-set channel standard deviations


def build_transform(kind: str, randaugment=(2, 9)):
    """kind = 'train' | 'gcsc' | 'eval'."""
    to_tensor = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if kind == "eval":
        return transforms.Compose(to_tensor)
    geometric = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if kind == "train":
        return transforms.Compose(geometric + to_tensor)
    if kind == "gcsc":
        num_ops, magnitude = randaugment
        return transforms.Compose(geometric + [transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)]
                                  + to_tensor)
    raise ValueError(f"unknown transform kind {kind!r}")


def cifar10(root, train: bool, transform, download: bool = False):
    """torchvision's CIFAR-10. `train=True` is the official 50k training partition (we split it 45k/5k);
    `train=False` is the official 10k test set (final known-class evaluation only)."""
    return datasets.CIFAR10(root=root, train=train, transform=transform, download=download)


def load_split(split_file):
    """The committed 45k/5k indices into the official training partition (see make_splits.py)."""
    s = load_json(split_file)
    return s["train"], s["val"]


def known_split(root, split_file, split: str, transform, debug_subset: int = 0):
    """One of the three KNOWN splits as a Dataset:
        'train' - 45,000 images of the official training partition (optimisation, Mahalanobis statistics)
        'val'   -  5,000 images of the official training partition (checkpoint selection, thresholds)
        'test'  - the official 10,000-image test set (final known-class evaluation)
    `debug_subset` > 0 keeps only the first N images (smoke tests only - never for reported results)."""
    if split in ("train", "val"):
        tr, va = load_split(split_file)
        idx = tr if split == "train" else va
        ds = Subset(cifar10(root, True, transform), idx)
    elif split == "test":
        base = cifar10(root, False, transform)
        ds = Subset(base, list(range(len(base))))
    else:
        raise ValueError(split)
    if debug_subset:
        ds = Subset(ds.dataset, list(ds.indices)[:debug_subset])
    return ds


def labels_of(subset) -> np.ndarray:
    """Labels of a Subset without decoding any image."""
    return np.asarray(subset.dataset.targets)[np.asarray(subset.indices)]


def make_loader(ds, batch_size, shuffle, num_workers, seed, drop_last=False):
    """A DataLoader with its OWN random stream (make_generator(seed)) so the shuffling order and the
    augmentation randomness depend only on the seed, not on anything else that ran before.
    persistent_workers keeps the worker processes alive across epochs (faster; same random streams)."""
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
              pin_memory=torch.cuda.is_available(), generator=make_generator(seed),
              worker_init_fn=seed_worker)
    if num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw)
