"""ResNet-18 used by Tasks 2 and 3 + the frozen-BatchNorm-statistics policy.

The network is split into F (backbone -> 512-d feature) and C (7-way linear head),
because every alignment method acts on F(x), the feature right before the head.
"""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.feat_dim = net.fc.in_features            # 512
        net.fc = nn.Identity()
        self.backbone = net                           # F
        self.head = nn.Linear(self.feat_dim, num_classes)   # C (new, randomly initialised)

    def forward(self, x, return_features: bool = False):
        f = self.backbone(x)
        logits = self.head(f)
        return (f, logits) if return_features else logits


def freeze_bn_running_stats(model: nn.Module) -> None:
    """BATCHNORM POLICY (Tasks 2 & 3).
    In train mode BN normalises with *batch* statistics and updates running means with
    them. With mixed source+target batches that would itself be a form of adaptation
    (like AdaBN) and would confound the comparison of alignment losses. Calling .eval()
    on BN modules ONLY makes them use -- and never update -- the ImageNet running stats.
    gamma/beta are ordinary parameters and still receive gradients.
    Call this every time right AFTER model.train()."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def set_train_mode(model: nn.Module) -> None:
    model.train()
    freeze_bn_running_stats(model)
