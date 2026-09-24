"""CIFAR-100 UNKNOWNS - evaluation only, and only after everything is frozen.

Handout: "Unknowns come from the following fixed CIFAR-100 test classes ... Each group contains 800 images.
The grouping is fixed and may not be revised after seeing results. CIFAR-100 training images may not be used
for training, checkpoint selection, score design, or threshold selection."

Three guards live here:
  1. Only the CIFAR-100 TEST partition is ever loaded (`train=False` is hard-coded; there is no argument
     that could switch it). torchvision computes an MD5 checksum of the training file when the dataset object
     is built, but its images are never unpickled or used.
  2. `load_unknowns` refuses to run unless the freeze record exists and the checkpoints, score code and
     thresholds on disk still match it (task4/audit.py). So no unknown image can be loaded while anything it
     could influence is still changeable.
  3. The first time the unknowns are loaded, the time is appended to an access log next to the freeze
     record. The log shows that the first access came after the freeze.

The near/far grouping below is copied verbatim from the handout.
"""
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets

NEAR = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]
GROUP_CODE = {"near": 1, "far": 2}           # the known CIFAR-10 test images get group code 0


class UnknownSet(Dataset):
    """The 1,600 fixed unknown images: 8 near + 8 far classes x 100 CIFAR-100 test images each.

    Items are (image, group_code) with group_code 1 = near, 2 = far. The metadata arrays `index` (position in
    the CIFAR-100 test set), `fine_class` and `group` are kept alongside for the failure analysis."""

    def __init__(self, root, transform):
        base = datasets.CIFAR100(root=root, train=False, transform=None, download=False)   # TEST split only
        name_to_id = {n: i for i, n in enumerate(base.classes)}
        missing = [c for c in NEAR + FAR if c not in name_to_id]
        assert not missing, f"CIFAR-100 class names not found: {missing}"
        targets = np.asarray(base.targets)
        index, fine, group = [], [], []
        for g, names in (("near", NEAR), ("far", FAR)):
            for name in names:
                idx = np.flatnonzero(targets == name_to_id[name])
                assert len(idx) == 100, f"{name}: expected 100 CIFAR-100 test images, found {len(idx)}"
                index += idx.tolist(); fine += [name] * len(idx); group += [g] * len(idx)
        self.base, self.transform = base, transform
        self.index, self.fine_class, self.group = np.asarray(index), np.asarray(fine), np.asarray(group)
        n_near, n_far = int((self.group == "near").sum()), int((self.group == "far").sum())
        assert n_near == 800 and n_far == 800, f"expected 800 near + 800 far, got {n_near} + {n_far}"

    def __len__(self):
        return len(self.index)

    def raw_image(self, i) -> np.ndarray:
        """The original 32x32x3 uint8 image (for the failure-analysis figure)."""
        return self.base.data[self.index[i]]

    def __getitem__(self, i):
        img, _ = self.base[self.index[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, GROUP_CODE[self.group[i]]


def load_unknowns(root, transform, frozen_file):
    """The ONLY way the pipeline obtains CIFAR-100 images. Checks the freeze record first."""
    from task4.audit import log_unknown_access, verify_frozen   # local import: audit imports nothing from here
    verify_frozen(frozen_file)                 # raises if anything is missing or changed since the freeze
    ds = UnknownSet(root, transform)
    log_unknown_access(frozen_file, n_images=len(ds))
    return ds


def download_cifar100(root):
    """Fetch the CIFAR-100 archive (the notebook calls this only in the evaluation Part, after the freeze).
    Downloading is not loading: the archive is fetched and unpacked, but no image is decoded here."""
    from torchvision.datasets.utils import download_and_extract_archive
    Path(root).mkdir(parents=True, exist_ok=True)
    folder = Path(root) / datasets.CIFAR100.base_folder
    if all((folder / f).exists() for f in ("train", "test", "meta")):   # torchvision MD5-checks all three
        print(f"CIFAR-100 already present under {root}")
        return
    C = datasets.CIFAR100
    download_and_extract_archive(C.url, str(root), filename=C.filename, md5=C.tgz_md5)
