"""PACS dataset access shared by Tasks 2 and 3.

Expected layout (the 'kfold' folder distributed with PACS / DomainBed):
    data/PACS/{photo,art_painting,cartoon,sketch}/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg|png
Download with:  python -m shared.download_pacs --root data
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights

SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN = "sketch"
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

_W = ResNet18_Weights.IMAGENET1K_V1.transforms()     # normalisation tied to the pretrained weights
MEAN, STD = _W.mean, _W.std


def scan_domain(root, domain):
    """Sorted list of [relative_path, label] for one domain (sorted -> deterministic)."""
    items = []
    for label, cls in enumerate(CLASSES):
        folder = Path(root) / domain / cls
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _EXTS)
        items += [[str(p.relative_to(root)), label] for p in files]
    if not items:
        raise FileNotFoundError(f"No images under {Path(root) / domain}")
    return items


def train_transform():
    # Resize to 256x256 (square, as specified), random 224 crop + horizontal flip.
    return T.Compose([T.Resize((256, 256)), T.RandomCrop(224), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform():
    return T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class PACSImages(Dataset):
    """return_labels=False is used for the UNLABELED target set in Task 2: the class label
    never even leaves the dataset object, which makes target-label leakage impossible
    in the training loop by construction."""

    def __init__(self, root, items, transform, return_labels=True, domain_id=None):
        self.root, self.items, self.transform = Path(root), items, transform
        self.return_labels, self.domain_id = return_labels, domain_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        x = self.transform(Image.open(self.root / rel).convert("RGB"))
        return (x, label) if self.return_labels else x
