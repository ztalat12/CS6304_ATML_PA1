"""Step 3 - GCSC: the Vanilla recipe with exactly one change, RandAugment(num_ops=2, magnitude=9).

Vaze et al. (2022) argue that a better closed-set classifier also separates known from unknown better when it
is scored with the Maximum Logit Score (MLS). Their "good closed-set classifier" used several training tricks;
the handout isolates ONE of them - stronger positive augmentation - so that any change in rejection can be
attributed to the augmentation alone:

    "Train a second ten-class ResNet-18 using exactly the vanilla recipe with one change: insert
     RandAugment(num_ops=2, magnitude=9) after the crop and flip and before conversion and normalization."

RandAugment applies 2 randomly chosen operations per image (rotate, shear, colour, contrast, posterize, ...)
at magnitude 9 of 31 bins. Everything else - model, random initialisation (same seed, so the SAME starting
weights as Vanilla), optimiser, schedule, epochs, checkpoint rule - is inherited from Vanilla unchanged.
"""
from task4.data.cifar10 import build_transform
from task4.methods.vanilla import Vanilla


class GCSC(Vanilla):
    name = "gcsc"

    def __init__(self, randaugment_num_ops: int = 2, randaugment_magnitude: int = 9, seed: int = 6304):
        super().__init__(seed)
        self.ra = (randaugment_num_ops, randaugment_magnitude)

    def train_transform(self):
        return build_transform("gcsc", randaugment=self.ra)

    def describe(self):
        return (f"Vanilla + RandAugment(num_ops={self.ra[0]}, magnitude={self.ra[1]}) after crop/flip, "
                "before ToTensor/Normalize")
