"""Frozen backbones for Task 1.

Each wrapper takes a [0,1] canvas, applies ITS OWN normalisation, and returns the
representation the handout asks for:
  * ResNet-50 (IMAGENET1K_V2): 2048-d global-average-pooled feature (fc -> Identity)
  * ViT-B/16  (IMAGENET1K_V1): 768-d final class token after the last LayerNorm
                               (torchvision's `heads` -> Identity)
  * CLIP ViT-B-32 (OpenAI):    512-d image embedding, L2-normalised

"Frozen" = eval mode (no dropout, fixed norm statistics) + requires_grad=False.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

BACKBONES = ("resnet50", "vit_b16", "clip_b32")


class FrozenBackbone(nn.Module):
    feat_dim: int = 0

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()

    def train(self, mode: bool = True):          # a frozen backbone never leaves eval mode
        return super().train(False)

    def encode(self, x_normalised):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.encode((x01 - self.mean) / self.std).float()


class ResNet50Backbone(FrozenBackbone):
    feat_dim = 2048

    def __init__(self, pretrained: bool = True):
        w = ResNet50_Weights.IMAGENET1K_V2
        t = w.transforms()                         # read normalisation from the weights
        super().__init__(t.mean, t.std)
        net = resnet50(weights=w if pretrained else None)
        net.fc = nn.Identity()                     # output = GAP feature
        self.net = net
        self._freeze()

    def encode(self, x):
        return self.net(x)


class ViTB16Backbone(FrozenBackbone):
    feat_dim = 768

    def __init__(self, pretrained: bool = True):
        w = ViT_B_16_Weights.IMAGENET1K_V1
        t = w.transforms()
        super().__init__(t.mean, t.std)
        net = vit_b_16(weights=w if pretrained else None)
        # torchvision: x = encoder(x) (ends with LayerNorm); x = x[:, 0]; x = heads(x)
        net.heads = nn.Identity()                  # output = final class token
        self.net = net
        self._freeze()

    def encode(self, x):
        return self.net(x)


class CLIPB32Backbone(FrozenBackbone):
    feat_dim = 512

    def __init__(self, pretrained: bool = True):
        import open_clip
        super().__init__(open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD)
        # The OpenAI checkpoint was trained with QuickGELU; forcing it avoids a silent
        # activation mismatch (open_clip warns about this for 'ViT-B-32' + 'openai').
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai" if pretrained else None, force_quick_gelu=True)
        self.net = model
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._freeze()

    def encode(self, x):
        return F.normalize(self.net.encode_image(x).float(), dim=-1)   # unit-norm embedding

    @torch.no_grad()
    def text_classifier(self, classnames, template: str = "a photo of a {}.") -> torch.Tensor:
        """Zero-shot 'weights': one normalised text embedding per class prompt."""
        tokens = self.tokenizer([template.format(c) for c in classnames]).to(self.mean.device)
        return F.normalize(self.net.encode_text(tokens).float(), dim=-1)

    @torch.no_grad()
    def zero_shot_logits(self, img_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """CLIP logits = learned temperature (logit_scale ~ 100) x cosine similarity.
        Softmax over these 'scaled class similarities' gives zero-shot confidence."""
        scale = self.net.logit_scale.exp().float().to(img_feats.device)
        return scale * img_feats @ text_feats.to(img_feats.device).T


def build_backbone(name: str, device, pretrained: bool = True) -> FrozenBackbone:
    cls = {"resnet50": ResNet50Backbone, "vit_b16": ViTB16Backbone, "clip_b32": CLIPB32Backbone}[name]
    return cls(pretrained=pretrained).to(device)
