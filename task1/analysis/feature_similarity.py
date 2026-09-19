"""Representation stability:  I_T = 1/N sum cos( f(x_i), f(T(x_i)) ).

Why cosine? All three representations are compared by direction (CLIP embeddings are
unit-norm by construction; linear heads are mostly sensitive to direction too), and
cosine is invariant to the global feature scale, which differs wildly between backbones.

IMPORTANT: absolute cosine values are NOT comparable across backbones (a ViT's CLS
tokens can all live in a narrow cone, so even unrelated images have cos ~ 0.5). Compare
each backbone's intervention values against *its own* baseline, e.g. the mean cosine
between two random different clean images, which we also report ('random_pair_baseline').
"""
import numpy as np
import torch
import torch.nn.functional as F


def cosine_per_sample(f_clean: torch.Tensor, f_trans: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(f_clean.float(), f_trans.float(), dim=1)


def stability(f_clean, f_trans) -> dict:
    c = cosine_per_sample(f_clean, f_trans)
    return {"mean": float(c.mean()), "std": float(c.std()), "n": len(c)}


def random_pair_baseline(f_clean: torch.Tensor, seed: int = 6304) -> float:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(f_clean), generator=g)
    same = perm == torch.arange(len(f_clean))
    perm[same] = (perm[same] + 1) % len(f_clean)
    return float(cosine_per_sample(f_clean, f_clean[perm]).mean())


def stability_vs_prediction(f_clean, f_trans, pred_clean, pred_trans) -> dict:
    """Links the two levels of analysis (Research Question 3): is the representation
    shift larger for images whose prediction flipped than for those that stayed?"""
    c = cosine_per_sample(f_clean, f_trans).numpy()
    same = np.asarray(pred_clean) == np.asarray(pred_trans)
    return {"cos_pred_unchanged": float(c[same].mean()) if same.any() else float("nan"),
            "cos_pred_changed": float(c[~same].mean()) if (~same).any() else float("nan"),
            "n_unchanged": int(same.sum()), "n_changed": int((~same).sum())}
