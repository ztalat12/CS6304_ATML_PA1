# Task 4 — Open-set recognition (CIFAR-10 known, CIFAR-100 unknown)

Ten CIFAR-10 classes are **known**. Sixteen fixed CIFAR-100 **test** classes are **unknown** and are used only for
final evaluation: 8 *near* (bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel) and 8 *far*
(bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe), 800 images each. Seed **6304** everywhere.

| Model | What changes | Selected by |
|---|---|---|
| **Vanilla** | CIFAR ResNet-18, cross-entropy, random init | CIFAR-10 val accuracy |
| **GCSC** | Vanilla + `RandAugment(num_ops=2, magnitude=9)` after crop/flip, before ToTensor/Normalize. Nothing else | CIFAR-10 val accuracy |
| **PROSER** | Vanilla checkpoint + 5 dummy classifiers, fine-tuned 50 epochs at lr 1e-3 with classifier placeholders (β = 1) and manifold-mixup data placeholders (γ = 0.1) | CIFAR-10 val accuracy (10 known logits) |

Scores, all as *unknownness* u(x) (larger = more unknown): **MSP** 1 − max softmax, **MLS** − max logit,
**Energy** − logsumexp, **Mahalanobis** min over classes of the diagonal-covariance distance,
**PROSER-detect** p(dummy) − max p(known) at temperature 1024.
Threshold for every model × score: **τ = 95th percentile of u on the CIFAR-10 validation split**. Accept if u ≤ τ.
The optional RPL extension was not done.

## How to run

The Kaggle notebook [`notebooks/task4_kaggle.ipynb`](../notebooks/task4_kaggle.ipynb) runs everything in order
(GPU T4, Internet ON, about 3 hours). The same steps from a shell, run from the repository root:

```bash
python -m task4.data.make_splits --data_root data --download       # once; the split file is committed
python -m task4.train --config task4/configs/vanilla.yaml --set data_root=data
python -m task4.train --config task4/configs/gcsc.yaml    --set data_root=data
python -m task4.train --config task4/configs/proser.yaml  --set data_root=data    # starts from vanilla/best.pt
python -m task4.extract_outputs --stage known --set data_root=data                 # CIFAR-10 only
python -m task4.freeze                                                              # thresholds + freeze record
# ---- CIFAR-100 is touched for the first time below this line ----
python -c "from task4.data.cifar100_unknowns import download_cifar100; download_cifar100('data')"
python -m task4.extract_outputs --stage unknown --set data_root=data               # refuses without the freeze
python -m task4.evaluate_osr
python -m task4.evaluation.failure_analysis --set data_root=data
# ---- separate, optional, re-runnable any time (reads only CSVs; no GPU, no CIFAR-100) ----
python -m task4.evaluation.apply_plausibility_map
```

## Files

```
task4/
  configs/          base.yaml (shared recipe) + vanilla / gcsc / proser.yaml
                    plausibility_map.yaml (SUGGESTED failure labels - separate, editable, not frozen)
                    proser_bncal.yaml (POST-HOC variant: PROSER + BatchNorm re-estimated each epoch)
  data/
    make_splits.py        stratified 90/10 split of the official 50k training set, seed 6304 -> 45,000 / 5,000
    splits/cifar10_split_seed6304.json   the committed indices
    cifar10.py            transforms (train / gcsc / eval), CIFAR-10 splits, seeded loaders
    cifar100_unknowns.py  the 16 fixed classes, TEST split only, loadable only after the freeze
  models/resnet_cifar.py  torchvision ResNet-18 with 3x3 stride-1 conv1 and no max-pool; split after layer2
  methods/                vanilla.py, gcsc.py, proser.py, manifold_mixup.py, bn_recalibration.py (post-hoc only)
  posthoc/                bn_diagnosis.py, evaluate_variant.py (post-hoc; see below)
  run1/                   key numbers recovered from run 1 (see "Runs")
  scores/                 msp.py, mls.py, energy.py, mahalanobis.py, proser_detection.py
  evaluation/             thresholds.py, metrics.py, stats.py, plots.py, failure_analysis.py, preregistered.py,
                          apply_plausibility_map.py
  train.py                one training loop for all three models
  extract_outputs.py      saves features + logits (stage "known", then stage "unknown")
  freeze.py               Mahalanobis statistics, thresholds, freeze record (CIFAR-10 train/val only)
  audit.py                SHA-256 checks behind the freeze record; CIFAR-100 access log
  evaluate_osr.py         tables, uncertainty, agreement, per-class results, figures
  cache/                  saved features/logits (git-ignored, large)
  results/
    runs/<model>/         config_resolved.json, history.json, summary.json (best.pt is git-ignored)
    frozen/               frozen.json, maha_<model>.npz, unknown_access_log.json
    scores/               per-image scores for val / test / unknown, every model
    final/                the tables, statistics, figures, failures/
```

## How the handout's leakage rule is enforced

*"CIFAR-100 training images may not be used for training, checkpoint selection, score design, or threshold
selection"*, and unknowns *"may be evaluated only after every checkpoint and score definition has been fixed"*.

1. **Order.** Train all three models, then extract CIFAR-10 outputs, then `task4.freeze`, and only then CIFAR-100.
   The notebook does not even download CIFAR-100 before the freeze.
2. **Freeze record** (`results/frozen/frozen.json`): the SHA-256 of every selected checkpoint, of every file that
   defines a score, the threshold rule, the features/logits themselves (network, eval transform, extraction code)
   or the unknown grouping, and of the Mahalanobis statistics. It also holds
   the thresholds, the score definitions, and the pre-registered tables, comparisons and failure-selection rule
   (`evaluation/preregistered.py`). Everything in it comes from CIFAR-10 train/val only.
3. **Guard.** CIFAR-100 can only be loaded through `cifar100_unknowns.load_unknowns`, which re-verifies every hash
   first and stops if anything changed. It loads only the CIFAR-100 **test** split. torchvision computes an MD5
   checksum of the training file when it builds the dataset object, but those images are never unpickled or used.
4. **Access log** (`results/frozen/unknown_access_log.json`): a timestamp for every CIFAR-100 load. The first one
   comes after `frozen_at`. `freeze.py` refuses to re-freeze changed state after an access unless a reason is
   given. The old record and log are then archived with that reason under `results/frozen/superseded/`.
5. **Traceability.** Every reported number is recomputed from `results/scores/*.csv`, which hold the per-image
   scores, and from the frozen thresholds.

## Failure analysis: selection is fixed, labelling is a judgement

The handout asks to record, for at least three near and three far unknowns accepted under the Vanilla-MLS
threshold, the unknown class, predicted class, score and threshold. It also asks to *distinguish semantically
plausible confusions from surprising failures*. Selection and labelling are kept apart.

1. **Selection**, by `evaluation/failure_analysis.py --view ...`. Every rule is pre-registered in the freeze
   record:
   - `main` (the required evidence): the most confidently accepted image of each unknown class, first 6 per
     group, at least 3. Written to `results/final/failures/`.
   - `confident_borderline` (supplementary view C): per class, the most confidently accepted image and the
     borderline one (the accepted image closest to τ). Written to `failures/view_confident_borderline/`.
   - `random` (supplementary view B): 6 accepted images per group, sampled with seed 6304, to show typical
     failures. Written to `failures/view_random_sample/`.

   None of these outputs carry plausible/surprising labels.
2. **Your judgement.** `failures/your_judgement.csv` holds one row per selected image across all views, with an
   `in_views` column and blank `your_call` (plausible / surprising) and `your_reason` columns. Rows are only ever
   added, so filled-in entries are never overwritten.
3. **Suggested map, separate.** `configs/plausibility_map.yaml` is a suggestion: vehicles → automobile/truck,
   wolf → dog, fox → dog/cat, leopard → cat, camel → horse/deer, far → none.
   - It is not pre-registered and changes no metric.
   - `evaluation/apply_plausibility_map.py` applies it to all views and writes into its own folder,
     `failures/suggested_map/`.
   - Once the sheet is filled, it also writes `judgement_vs_map.csv`, comparing your calls with the map row by
     row.
   - Edit the YAML and re-run as often as needed; the script reads only the saved CSVs.

## Runs

- **Run 1** (20 Sep 2026, interactive Kaggle session) finished without errors, but the session ended before its
  output zips were downloaded. Its executed notebook is `notebooks/task4_kaggle_run1_executed.ipynb`, and
  `run1/run1_key_numbers.json` holds what it printed: checkpoint hash prefixes, tables and training logs.
- **Run 2** (a committed Kaggle version) is the run whose files are in `results/`. Nothing that affects the three
  pre-registered models, scores or thresholds changed between the runs: same code and seed. The only difference in
  `train.py` is that it also writes `last.pt`, which changes neither training nor `best.pt`. The notebook prints
  run 2 next to run 1:
  - Part E2 compares the checkpoint hashes, which should be bit-identical if training was deterministic.
  - Part F2 compares the tables.

  Reported numbers come from run 2; run 1 is a reproducibility check.

## Post-hoc: PROSER and BatchNorm (not part of the pre-registered evaluation)

**What run 1 showed** (CIFAR-10 validation only):
- PROSER's validation accuracy fell from 94.6% after its first fine-tuning epoch to 85–93%.
- Training accuracy stayed at 99.9%.
- The strongest dummy came to beat every known class on up to 100% of validation images.
- The checkpoint rule therefore selected epoch 1, so the pre-registered PROSER rows describe Vanilla plus one
  epoch of PROSER.

**Suspected mechanism.** The data-placeholder pass sends manifold-mixed layer2 features through layer3/layer4 in their
own forward call. BatchNorm's running statistics, which are used only in evaluation mode, therefore also average
over mixed features. The reference code has the same structure.

**Post-hoc additions** (Part I of the notebook, all outputs in `results/posthoc/`):
1. `posthoc/bn_diagnosis.py` (CIFAR-10 only) re-estimates the running statistics of saved checkpoints from real
   training images and re-measures validation accuracy and dummy-wins. It also reports which network stage's
   statistics were off; mixed features only reach layer3 and layer4. Vanilla is the control.
2. `configs/proser_bncal.yaml`: PROSER trained exactly as before, except that BatchNorm statistics are re-estimated
   from real training images after every epoch, before validation and checkpoint selection. Running statistics do
   not influence training, so the weights follow the same path. Only the evaluation-time statistics, and
   therefore which epoch is selected, change.
3. `posthoc/evaluate_variant.py` writes a separate freeze record (`results/posthoc/frozen_posthoc.json`: variant
   checkpoint hash, post-hoc code hashes, thresholds, pointer to the main record) before the variant's first
   CIFAR-100 access. It then compares the variant with Vanilla-MLS and the pre-registered PROSER rows on the same
   images, with bootstrap CIs, McNemar tests and Holm correction within this post-hoc family.

The variant was decided after run 1's CIFAR-100 results had been seen, so it is reported as post-hoc, next to (never
instead of) the pre-registered PROSER.

## Implementation notes (where a choice had to be made)

- **Architecture.** torchvision `resnet18(weights=None)`, with `conv1 = Conv2d(3, 64, 3, stride 1, padding 1)`
  (Kaiming-normal, like every other conv) and `maxpool = Identity`. f(x) is the 512-d global-average-pooled feature.
- **Split.** A NumPy `default_rng(6304)` permutation inside each class, taken in class order; the first 10% go to
  validation. The indices are saved, so the split no longer depends on library versions.
- **Schedule.** `CosineAnnealingLR(T_max = epochs)`, stepped once per epoch. Weight decay applies to all parameters.
- **Checkpoint.** Highest validation accuracy; a tie keeps the earlier epoch. For PROSER, the epochs are those of
  the fine-tuning (1–50), so the selected model always has trained dummies.
- **PROSER loss** (paper formulation, handout weights). The logit vector is f̂ = [10 known logits, max over the
  5 dummies]. In the first half of the batch: `CE(f̂, y) + 1.0 · CE(f̂ with z_y masked, dummy)`. In the second
  half: mix layer2 feature maps of **different-class** pairs with λ ~ Beta(2, 2), one λ per batch as in the
  reference code, then `0.1 · CE(f̂(mix), dummy)`. Same-class pairs from the random permutation are dropped, about
  10% of them.
  Where each setting comes from:
  - **Maximum over the dummies.** Paper Eq. 4, f̂(x) = [Wᵀφ(x), max_k ŵ_kᵀφ(x)] ("only considers the nearest dummy
    classifier"). The handout defers to the paper for "handling of the dummy classifiers" and speaks of "the
    strongest dummy response".
  - **Different-class pairs only.** The handout states y_i ≠ y_j, and so does the paper's mixup equation.
  - **Half-batch order.** The handout: first half classifier placeholders, second half data placeholders. The
    paper's Algorithm 1 uses the same order.
  - **One λ per training iteration.** The handout only says λ ~ Beta(2, 2). The paper's Algorithm 1 and the
    reference code both draw λ once per iteration (mini-batch).
  - **Calibration.** The paper's Algorithm 1 calibrates the dummy on validation so that 95% of known images are
    accepted. Here the validation 95th-percentile threshold τ does the same job, as the handout requires for every
    score.

  Differences from the reference code (`traindummy`), none of which depart from the paper:
  - The reference's default is 1 dummy. With several dummies, its known-class CE term and its mixup term use all
    the dummies (the mixup target is the *first* dummy), and only the masked term uses their maximum. We take the
    maximum everywhere, as the paper does. With one dummy the two are identical.
  - It mixes the *first* half of the batch; we follow the handout's order.
  - It hard-codes 0.01 for the mixup term and uses a constant learning rate with Beta(1, 1). We use the handout's
    γ = 0.1, cosine decay and Beta(2, 2).
  - It uses a plain `randperm` without removing same-class pairs. We drop them, following the paper and the
    handout (y_i ≠ y_j).
- **PROSER detection.** Exactly the reference `valdummy` / `CONF_DeltaP` score: softmax of [known, max dummy] / 1024,
  then p_dummy − max p_known. The reference's bias search is over `[0]`, so there is no bias.
  - The reference reports `max(AUROC, 1 − AUROC)`, whichever direction is better. We fix the direction:
    larger = more unknown.
  - The paper calibrates a bias on the raw gap (max dummy − max known). At T = 1024 the ΔP score ranks images
    almost exactly like that gap.
  - Calibration uses the common rule for every score: the 95th-percentile validation threshold.
- **MSP numerics.** 1 − max p is computed as r/(1 + r), where r is the sum of exp(z_k − z_max) over the non-max
  classes. This is the same quantity without float32 rounding to exactly 0, which would create artificial ties.
- **Mahalanobis.** Pooled within-class variance divided by N (the maximum-likelihood estimate, as in Lee et al.'s
  empirical covariance), plus 1e-6, from the unaugmented 45k training features.
- **FPR@95TPR** is the % of unknowns accepted at the validation-calibrated τ, following the handout's convention.
  It is not re-tuned on the test set.
- **Uncertainty.**
  - AUROC 95% CIs and paired AUROC differences come from a stratified bootstrap (B = 2000, seed 6304).
  - Rates have Wilson intervals.
  - Paired rejection and CSA differences use exact McNemar tests.
  - Holm correction is applied within each pre-registered family.
- **Engineering, same maths.**
  - Mixed precision (AMP) and channels-last memory layout for speed on the GPU.
  - Deterministic cuDNN, 4 data-loader workers fixed in the config, and every random stream seeded. The global
    average pool is a plain mean, whose backward pass is deterministic. GPU training is still not guaranteed to be
    bit-for-bit repeatable: `torch.use_deterministic_algorithms` is not switched on, and some CUDA backward
    kernels use atomic additions. The selected checkpoints, the saved outputs and the freeze record are what the
    reported numbers trace to.
  - Validation, extraction and every score run in float32/float64.
  - `extract_outputs` checks that the saved validation logits reproduce the checkpoint's recorded validation
    accuracy.

## External code and ideas

| Component | Source |
|---|---|
| ResNet-18 definition, RandAugment | torchvision |
| PROSER losses and detection score | Zhou et al., *Learning Placeholders for Open-Set Recognition*, CVPR 2021; reference code [zhoudw-zdw/CVPR21-Proser](https://github.com/zhoudw-zdw/CVPR21-Proser) (`proser_unknown_detection.py`), re-implemented here |
| MLS, GCSC idea | Vaze et al., *Open-Set Recognition: a Good Closed-Set Classifier is All You Need?*, ICLR 2022 |
| MSP | Hendrycks & Gimpel, ICLR 2017 |
| Energy | Liu et al., NeurIPS 2020 |
| Mahalanobis | Lee et al., NeurIPS 2018 (handout variant: shared diagonal covariance) |

Code was written with the help of an LLM (Claude). The author has reviewed it and is responsible for it.
