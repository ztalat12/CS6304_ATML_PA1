"""STL-10 access for Task 1.

KEY DESIGN RULE (from the handout): every intervention is built on ONE common
224x224 RGB image in [0, 1], and only afterwards does each model apply its own
normalisation. That way ResNet, ViT and CLIP see *exactly* the same pixels, and any
difference in behaviour comes from the models, not from different preprocessing.

STL-10 images are 96x96, so we upsample them once with bicubic interpolation.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import STL10

STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]


def load_stl10_arrays(root: str, split: str):
    """Return (uint8 array N x 3 x 96 x 96, int64 labels 0..9, class names)."""
    ds = STL10(root=root, split=split, download=True)
    # torchvision already stores STL-10 as channel-first uint8 and shifts labels to 0..9
    return ds.data, np.asarray(ds.labels, dtype=np.int64), list(ds.classes)


def to_common_canvas(batch_uint8, size: int = 224) -> torch.Tensor:
    """uint8 (B,3,h,w) -> float (B,3,size,size) in [0,1] (the shared canvas)."""
    x = torch.as_tensor(np.asarray(batch_uint8)).float().div_(255.0)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return x.clamp_(0.0, 1.0)


def canvas_to_uint8(x01: torch.Tensor) -> torch.Tensor:
    """Store canvases compactly (500 images x 224^2 x 3 = 75 MB as uint8)."""
    return (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)
