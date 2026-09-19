"""AdaIN style transfer (Huang & Belongie, 2017).

ATTRIBUTION: the two network definitions below are copied verbatim (layer-for-layer)
from naoto0804/pytorch-AdaIN (net.py, MIT licence) so that its released weights
`vgg_normalised.pth` and `decoder.pth` load without modification:
    https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0

HOW ADAIN WORKS
  1. Encode content c and style s with a fixed VGG-19 up to relu4_1.
  2. Re-normalise the content feature map channel-wise so that its per-channel mean and
     standard deviation equal those of the style feature map:
         AdaIN(x, y) = sigma(y) * (x - mu(x)) / sigma(x) + mu(y)
     Channel statistics summarise *texture/colour* (they are spatially averaged, so the
     WHERE information -- i.e. shape/layout -- is removed from them), while the spatial
     pattern of x keeps the *content layout*.
  3. A trained decoder maps the result back to pixels.
  alpha in [0,1] interpolates in feature space between content and stylised features:
  lower alpha = weaker texture cue, stronger shape cue.
"""
import torch
import torch.nn as nn

decoder_arch = lambda: nn.Sequential(  # noqa: E731
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)


def vgg_arch():
    layers = [nn.Conv2d(3, 3, (1, 1))]
    cfg = [(3, 64), (64, 64), "M", (64, 128), (128, 128), "M",
           (128, 256), (256, 256), (256, 256), (256, 256), "M",
           (256, 512), (512, 512), (512, 512), (512, 512), "M",
           (512, 512), (512, 512), (512, 512), (512, 512)]
    for c in cfg:
        if c == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True))
        else:
            layers += [nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(c[0], c[1], (3, 3)), nn.ReLU()]
    return nn.Sequential(*layers)


def calc_mean_std(feat, eps: float = 1e-5):
    N, C = feat.shape[:2]
    var = feat.reshape(N, C, -1).var(dim=2) + eps
    return feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1), var.sqrt().view(N, C, 1, 1)


def adaptive_instance_normalization(content_feat, style_feat):
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    return (content_feat - c_mean) / c_std * s_std + s_mean


class AdaIN(nn.Module):
    """Inputs and outputs are [0,1] RGB tensors (the released VGG's first 1x1 conv
    performs its own normalisation, so no ImageNet mean/std is applied here)."""

    # indices (in the full VGG) right after relu1_1, relu2_1, relu3_1, relu4_1
    SLICES = [(0, 4), (4, 11), (11, 18), (18, 31)]

    def __init__(self, vgg_path: str, decoder_path: str):
        super().__init__()
        vgg = vgg_arch()
        vgg.load_state_dict(torch.load(vgg_path, map_location="cpu"))
        self.encoder = nn.Sequential(*list(vgg.children())[:31])      # up to relu4_1
        self.decoder = decoder_arch()
        self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def multi_level_stats(self, x):
        """Channel mean/std at relu1_1..relu4_1 -- the 'style' description used by the
        automatic rejection rule (same statistics AdaIN's style loss matches)."""
        stats, h = [], x
        children = list(self.encoder.children())
        for a, b in self.SLICES:
            for layer in children[a:b]:
                h = layer(h)
            m, s = calc_mean_std(h)
            stats.append(torch.cat([m.flatten(1), s.flatten(1)], 1))
        return torch.cat(stats, 1)

    @torch.no_grad()
    def forward(self, content, style, alpha: float = 1.0):
        cf, sf = self.encoder(content), self.encoder(style)
        t = adaptive_instance_normalization(cf, sf)
        t = alpha * t + (1.0 - alpha) * cf
        out = self.decoder(t)
        return out[..., : content.shape[-2], : content.shape[-1]].clamp(0, 1)
