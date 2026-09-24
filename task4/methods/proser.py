"""Step 4 - PROSER: classifier placeholders + data placeholders (Zhou et al., CVPR 2021).

Idea. A closed-set classifier has no output that means "none of these". PROSER adds one: a dummy class, the
(K+1)-th output, and trains it with signals built ONLY from CIFAR-10 images, so that at test time an input that
matches no known class can land on the dummy instead of being forced into one of the ten classes.

Model. The selected Vanilla network plus five dummy classifiers (a Linear 512 -> 5). With several dummies the
paper takes their MAXIMUM as the single (K+1)-th response, so every loss below works on 11 numbers:
        f_hat(x) = [ z_1(x), ..., z_10(x),  max_k d_k(x) ]          (index 10 = the dummy)

Each mini-batch of 128 is split in two equal halves (handout):

  FIRST half - classifier placeholders (paper Eq. for L_CP, beta = 1):
        L_CP = CE(f_hat(x), y)  +  beta * CE(f_hat(x) with z_y removed, target = dummy)
    Term 1 keeps the true class the largest of all 11 outputs (known recognition is preserved).
    Term 2 masks the true-class logit and asks the dummy to beat the remaining nine known classes, so the
    dummy becomes the "second choice" of every known image. It learns to sit on the boundary around each
    class: an input that is not clearly class y should end up closer to the dummy than to another class.

  SECOND half - data placeholders (paper Eq. for L_DP, gamma = 0.1):
        h_tilde = lambda * phi_pre(x_i) + (1 - lambda) * phi_pre(x_j),  y_i != y_j,  lambda ~ Beta(2, 2)
        L_DP = CE(f_hat(h_tilde), target = dummy)       (f_hat computed from layer3 onwards)
    Mixed layer2 representations between two classes stand in for unknowns; they are pushed to the dummy.

  Total: L = L_CP + gamma * L_DP.

Relation to the reference code (github.com/zhoudw-zdw/CVPR21-Proser, proser_unknown_detection.py):
  * The reference defaults to ONE dummy; then its losses are exactly the ones above. With several dummies it
    concatenates all of them for the ordinary CE term and targets the FIRST dummy for the mixup term. We follow
    the paper's formulation instead (maximum over the dummies everywhere), which the handout asks for ("refer to
    the paper for the exact loss construction and handling of the dummy classifiers").
  * The reference mixes the first half and uses the second half for classifier placeholders; the handout
    specifies the opposite order, which we follow. Which half is which does not matter statistically (the
    batch is shuffled), but we match the handout literally.
  * Weights: the handout's beta = 1 and gamma = 0.1 (the reference script hard-codes 0.01 for the mixup term).
  * The reference computes the known and dummy heads with two separate forward passes; we compute the feature
    once and apply both heads (same maths, one pass).

Training accuracy and checkpoint selection use the ten KNOWN logits only (the handout: "Select the checkpoint
using CIFAR-10 validation accuracy only"; "compute CSA using only the ten known-class logits").
"""
import torch
import torch.nn.functional as F

from common.io import resolve
from task4.methods.manifold_mixup import MixupSampler, mix
from task4.methods.vanilla import Vanilla
from task4.models.resnet_cifar import CifarResNet18

MASK = -1e9          # "remove" a logit: exp(-1e9) = 0 in float32 (losses are computed in float32)


def with_dummy(z, d):
    """[10 known logits, max over the dummies] -> 11 numbers, in float32."""
    return torch.cat([z, d.max(dim=1, keepdim=True).values], dim=1).float()


class PROSER(Vanilla):
    name = "proser"

    def __init__(self, n_dummies=5, beta=1.0, gamma=0.1, mixup_alpha=2.0, mix_after="layer2",
                 detection_temperature=1024, seed=6304):
        super().__init__(seed)
        assert mix_after == "layer2", "the handout fixes manifold mixup after layer2, before layer3"
        self.n_dummies, self.beta, self.gamma = n_dummies, beta, gamma
        self.mixup = MixupSampler(alpha=mixup_alpha, seed=seed)
        self.detection_temperature = detection_temperature          # used by scores/, recorded here

    def build_model(self, cfg):
        path = resolve(cfg["init_from"])
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck.get("n_dummies", 0) == 0 and ck.get("method") == "vanilla", \
            f"{path} is not a Vanilla checkpoint - PROSER must start from the selected Vanilla model"
        model = CifarResNet18(num_classes=10)
        model.load_state_dict(ck["model"], strict=True)             # every Vanilla weight, exactly
        model.add_dummies(self.n_dummies)                           # 5 new, randomly initialised dummies
        print(f"PROSER initialised from {path} (Vanilla epoch {ck['epoch']}, val acc {ck['val_acc']:.2f}%) "
              f"+ {self.n_dummies} random dummy classifiers")
        return model

    def training_step(self, model, x, y):
        K = model.num_classes                                        # 10; index K of f_hat is the dummy
        half = x.size(0) // 2
        x1, y1 = x[:half], y[:half]                                  # classifier placeholders
        x2, y2 = x[half:], y[half:]                                  # data placeholders

        # ---- first half: classifier placeholders ----------------------------------------------------------
        z1, d1 = model.heads(model.features(x1))
        out1 = with_dummy(z1, d1)                                    # (half, 11)
        l_ce = F.cross_entropy(out1, y1)                             # true class wins over all 11
        masked = out1.scatter(1, y1.unsqueeze(1), MASK)              # remove the true-class logit
        dummy_t1 = torch.full_like(y1, K)
        l_dummy = F.cross_entropy(masked, dummy_t1)                  # dummy wins over the other nine

        # ---- second half: data placeholders (manifold mixup after layer2) ---------------------------------
        h = model.pre(x2)                                            # (n, 128, 16, 16)
        i, j = self.mixup.different_class_pairs(y2)
        lam = self.mixup.sample_lambda()
        if len(i) > 0:
            zm, dm = model.heads(model.post(mix(h, i, j, lam)))
            l_dp = F.cross_entropy(with_dummy(zm, dm), torch.full_like(i, K))
        else:                                                        # (practically impossible) no valid pair
            l_dp = out1.sum() * 0.0

        loss = l_ce + self.beta * l_dummy + self.gamma * l_dp
        logs = {"cp_ce": l_ce.detach(), "cp_dummy": l_dummy.detach(), "dp": l_dp.detach(),
                "pairs": torch.tensor(float(len(i))), "lam": torch.tensor(lam)}
        return loss, logs, z1.detach(), y1

    def describe(self):
        return (f"PROSER: {self.n_dummies} dummies (max = placeholder logit); first half CE + {self.beta} x "
                f"masked-CE->dummy; second half manifold mixup after layer2, lambda~Beta({self.mixup.alpha},"
                f"{self.mixup.alpha}), different classes only, CE->dummy weighted {self.gamma}")
