"""The CIFAR-appropriate ResNet-18 used by every Task 4 model.

Handout: "Replace the ImageNet 7x7, stride-2 first convolution with a 3x3, stride-1 convolution and remove the
initial max-pooling layer. Operate on the original 32x32 images."

WHY: the ImageNet stem (7x7 stride 2, then 3x3 max-pool stride 2) shrinks a 224x224 image by 4x before the
first residual block. On a 32x32 CIFAR image it would leave only 8x8 pixels for layer1 and 1x1 for layer4,
throwing away most spatial detail. With the CIFAR stem the feature maps are 32 -> 32 (layer1) -> 16 (layer2)
-> 8 (layer3) -> 4 (layer4) -> global average pool -> 512-d feature f(x) -> linear -> 10 logits z(x).

The network is split into two halves because PROSER needs to mix representations in the middle:
    pre(x)  = stem + layer1 + layer2      -> h (128 x 16 x 16)        "phi_pre" in the handout
    post(h) = layer3 + layer4 + avg pool  -> f (512)
    fc(f)   = the ten known-class logits
    dummy(f)= the PROSER dummy classifiers (only after add_dummies; absent for Vanilla and GCSC)
forward(x) == fc(post(pre(x))), so splitting changes nothing for Vanilla/GCSC.

Built from torchvision's resnet18(weights=None): random initialisation, as the handout requires for Vanilla
and GCSC. torchvision initialises every conv with Kaiming-normal (fan_out, relu); the replacement 3x3 conv1
gets the same initialiser so the whole network follows one initialisation procedure.
"""
import torch
import torch.nn as nn
import torchvision

FEATURE_DIM = 512


class CifarResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        net = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)   # 3x3, stride 1
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        net.maxpool = nn.Identity()                                                   # no initial max-pool
        self.net = net
        self.num_classes = num_classes
        self.n_dummies = 0
        self.dummy = None

    # --- PROSER ------------------------------------------------------------------------------------------
    def add_dummies(self, n: int) -> None:
        """Append n randomly initialised dummy classifiers (a Linear 512 -> n, PyTorch's default init)."""
        self.dummy = nn.Linear(FEATURE_DIM, n)
        self.n_dummies = n

    # --- the two halves --------------------------------------------------------------------------------------
    def pre(self, x):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.maxpool(x)                 # Identity
        return n.layer2(n.layer1(x))

    def post(self, h):
        n = self.net
        # global average pool as a mean over H and W: identical to torchvision's AdaptiveAvgPool2d((1, 1)) +
        # flatten, but its backward pass is deterministic on the GPU (AdaptiveAvgPool2d's uses atomic adds)
        return n.layer4(n.layer3(h)).mean(dim=(2, 3))

    def features(self, x):
        return self.post(self.pre(x))

    def forward(self, x):
        """Ten known-class logits only (what Vanilla/GCSC train on and what CSA/MLS use for every model)."""
        return self.net.fc(self.features(x))

    def heads(self, f):
        """(known logits, dummy logits or None) from a 512-d feature."""
        return self.net.fc(f), (self.dummy(f) if self.dummy is not None else None)

    def forward_all(self, x):
        """(feature, known logits, dummy logits or None) - what extract_outputs.py saves."""
        f = self.features(x)
        z, d = self.heads(f)
        return f, z, d


def load_model(ckpt_path, device="cpu"):
    """Rebuild a saved model (with its dummy classifiers, if it has them)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = CifarResNet18(num_classes=ck.get("num_classes", 10))
    if ck.get("n_dummies", 0):
        m.add_dummies(ck["n_dummies"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck
