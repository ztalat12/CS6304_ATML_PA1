"""Step 1 - Vanilla: a ten-class CIFAR ResNet-18 trained with plain cross-entropy from random initialisation.

Every method is a small class with the same four hooks, so train.py is shared and the ONLY differences
between runs are the ones the handout asks for:
    build_model(cfg)            -> the network to optimise
    train_transform()           -> the training augmentation
    training_step(model, x, y)  -> (loss, logs, logits for train accuracy, labels for train accuracy)
    describe()                  -> one line printed at the start of training
"""
import torch.nn.functional as F

from task4.data.cifar10 import build_transform
from task4.models.resnet_cifar import CifarResNet18


class Vanilla:
    name = "vanilla"

    def __init__(self, seed: int = 6304):
        self.seed = seed

    def build_model(self, cfg):
        return CifarResNet18(num_classes=10)            # random initialisation (seeded by train.py)

    def train_transform(self):
        return build_transform("train")                 # crop(32, pad 4) + flip

    def training_step(self, model, x, y):
        z = model(x)
        loss = F.cross_entropy(z.float(), y)            # loss in float32 even under mixed precision
        return loss, {"ce": loss.detach()}, z.detach(), y

    def describe(self):
        return "cross-entropy on the 10 CIFAR-10 classes; augmentation = random crop (pad 4) + horizontal flip"
