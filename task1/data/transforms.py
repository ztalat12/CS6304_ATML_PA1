"""Controlled interventions (all act on float canvases in [0,1], shape B x 3 x 224 x 224).

Each function changes ONE visual factor and tries to keep the others fixed:

  grayscale      removes chroma, keeps luminance and geometry
  hue_rotation   changes hue, keeps saturation/value (brightness) and geometry
  translate      moves the object, keeps pixels (reflection padding fills the border)
  patch_shuffle  destroys global arrangement, keeps every pixel and local texture
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ITU-R BT.601 luma weights (what PIL / torchvision use for RGB->L).
_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def grayscale(x: torch.Tensor) -> torch.Tensor:
    """Remove colour. We keep 3 channels because every backbone expects RGB input."""
    y = (x * _LUMA.to(x.device, x.dtype)).sum(1, keepdim=True)
    return y.expand(-1, 3, -1, -1).contiguous()


def hue_rotation(x: torch.Tensor, hue_factor: float = 0.5) -> torch.Tensor:
    """Rotate hue by hue_factor * 360 degrees (0.5 -> 180 deg: blue sky becomes orange,
    green grass becomes magenta). CHANGES: hue.  PRESERVES: saturation, HSV value
    (so edges/shading/geometry are intact), spatial layout.
    NOTE: perceived luminance is only approximately preserved because HSV 'value' is
    max(R,G,B), not luma -- mention this caveat when interpreting the result."""
    return TF.adjust_hue(x, hue_factor).clamp(0, 1)


# (dx, dy): positive dx moves the object RIGHT, positive dy moves it DOWN.
DIRECTIONS = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}


def translate(x: torch.Tensor, delta: int, direction: str) -> torch.Tensor:
    """Reflection-pad by `delta` then take a shifted HxW crop.

    Padded image P has the original at P[d:d+H, d:d+W]. To move the content by (dx,dy)
    we need out[r, c] = in[r-dy, c-dx] = P[r-dy+d, c-dx+d], i.e. the crop window starts
    at (d - dy, d - dx). Reflection (not zeros) avoids adding an artificial black border
    that a model could react to instead of the displacement itself.
    """
    if delta == 0:
        return x
    ux, uy = DIRECTIONS[direction]
    dx, dy = ux * delta, uy * delta
    H, W = x.shape[-2:]
    padded = F.pad(x, (delta, delta, delta, delta), mode="reflect")
    top, left = delta - dy, delta - dx
    return padded[..., top:top + H, left:left + W]


def make_patch_permutations(n_images: int, grid: int = 4, seed: int = 6304) -> np.ndarray:
    """One NON-identity permutation of grid*grid patches per image, from ONE seeded
    stream. The array is saved, so every model sees exactly the same shuffles."""
    rng = np.random.RandomState(seed)
    ident = np.arange(grid * grid)
    perms = np.empty((n_images, grid * grid), dtype=np.int64)
    for i in range(n_images):
        p = rng.permutation(grid * grid)
        while np.array_equal(p, ident):     # the handout requires a non-identity shuffle
            p = rng.permutation(grid * grid)
        perms[i] = p
    return perms


def patch_shuffle(x: torch.Tensor, perms: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """Split each image into grid x grid patches and reassemble them as perms[i]
    (output slot k receives input patch perms[i, k]). Pixel-space grid, independent of
    any architecture's token size -- we test reliance on global arrangement."""
    B, C, H, W = x.shape
    ph, pw = H // grid, W // grid
    x = x[..., : ph * grid, : pw * grid]
    # (B, C, g, g, ph, pw) -> (B, g*g, C, ph, pw)
    patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, grid * grid, C, ph, pw)
    idx = perms.to(x.device).long().view(B, grid * grid, 1, 1, 1).expand_as(patches)
    shuffled = torch.gather(patches, 1, idx)
    # back to (B, C, H, W)
    shuffled = shuffled.view(B, grid, grid, C, ph, pw).permute(0, 3, 1, 4, 2, 5)
    return shuffled.reshape(B, C, ph * grid, pw * grid)
