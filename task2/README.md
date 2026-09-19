# Task 2 — Unsupervised Domain Adaptation on PACS

**Setup.** Sources: Photo, Art Painting, Cartoon (labelled). Target: Sketch, unlabelled during training.
The protocol is transductive: all Sketch images are used without labels, and Sketch labels are read
only by `task2/evaluate_final.py`, after every checkpoint and setting is frozen.

**Methods compared** (ResNet-18, full fine-tuning, identical pipeline):
1. Source-only ERM (this checkpoint is also Task 3's ERM baseline)
2. DAN (MMD alignment)
3. DANN (adversarial alignment)
4. CDAN (class-conditional adversarial alignment)

## 0. Data and splits (once, from the repository root)

```bash
python -m shared.download_pacs --root data          # -> data/PACS/{photo,art_painting,cartoon,sketch}/<class>/
python -m shared.pacs_protocol --root data/PACS     # -> shared/splits/*.json (seed 6304)
```

The split files are committed. `pacs_sources_seed6304.json` holds the 80/20 split of each source
domain. `pacs_sketch_target.json` lists every Sketch image. The two are kept in separate files so
you can check that Task 3 never reads the Sketch list. On Colab, keep PACS on local disk and pass
`--set data_root=/content/data/PACS` (see `notebooks/task2_colab.ipynb`).

## 1. Train (checkpoints are selected on mean source-validation macro-F1 only)

```bash
python -m task2.train --config task2/configs/source_only.yaml
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml
# controlled study -- ONE family, chosen before any target evaluation (default: DAN lambda)
python -m task2.train --config task2/configs/study_dan_lambda0.1.yaml
python -m task2.train --config task2/configs/study_dan_lambda10.yaml
#   (alternative: study_dann_alpha0.25.yaml and study_dann_alpha0.5.yaml)
```

Each run writes the following to `task2/results/runs/<run_name>/`:
- `best.pt`: checkpoint, git-ignored
- `history.json`: per-epoch losses, discriminator accuracy, alpha, source-validation metrics
- `summary.json`
- `config_resolved.json`

## 2. Final evaluation — the only step that reads Sketch labels

```bash
python -m task2.evaluate_final                 # main comparison
python -m task2.evaluate_final --study dan     # or: --study dann
```

## 3. Where each piece of required evidence ends up (`task2/results/final/`)

| Handout requirement | File(s) |
|---|---|
| Main table: each source-val domain, mean source acc/F1, target acc/F1, Δ target acc, domain separability | `table_main.csv` |
| Loss / alignment / domain-loss curves | `figures/training_curves.png` (+ `runs/*/history.json`) |
| Per-class target accuracy changes | `per_class_target_acc.csv`, `figures/per_class_delta.png` |
| Dominant confusions, largest gain/drop per method | `class_analysis.json` |
| Failure cases (Source-only mistakes; correct→wrong after adaptation) | `figures/failures_*.png` |
| Controlled study | `table_study_<dan|dann>.csv`, `figures/study_<dan|dann>.png`, `figures/training_curves_study_*.png` |
| Audit record of which checkpoints were frozen when labels were first used | `frozen_main.json`, `frozen_<study>.json` |

## 4. File map

```
shared/pacs.py               PACS classes/domains, transforms, dataset (target labels never returned)
shared/pacs_protocol.py      the one split used by Tasks 2 and 3 (+ image-count check)
shared/download_pacs.py      DomainBed's PACS download
shared/models.py             ResNet-18 = backbone F (512-d) + head C; frozen-BatchNorm policy
shared/engine.py             the single training loop: balanced batches, AdamW, early stopping
shared/mmd.py                multi-kernel MMD (median heuristic), shared with Task 3 DAN-DG
shared/diagnostics.py        feature extraction + logistic-regression domain separability
task2/models/domain_discriminator.py   gradient reversal layer, alpha(p) schedule, discriminator
task2/methods/{source_only,dan,dann,cdan}.py   one loss per method
task2/train.py               trains one config
task2/evaluate_final.py      final Sketch evaluation, per-class analysis, figures
task2/evaluation/class_analysis.py     confusions, failure grids, curves, study plot
```

## Implementation choices to state in the report

- **MMD:**
  - Biased (V-statistic) estimate.
  - Kernel `exp(-d² / (m · median))` with m ∈ {0.5, 1, 2}.
  - The median is taken over off-diagonal pairwise squared distances in the combined batch and is not back-propagated.
- **DANN / CDAN:**
  - The gradient reversal layer and discriminator are trained in the same backward pass as the classifier.
  - α(p) uses p = step / (30 epochs × steps per epoch), so it follows the full budget even if early stopping ends training sooner.
  - The discriminator is optimised with the same AdamW settings as the network.
- **CDAN:** the discriminator input is `softmax(logits) ⊗ f` (3,584-d), with nothing detached and no entropy weighting, as the handout requires. The official CDAN code detaches the softmax.
- **Gradient clipping:** the global gradient norm is clipped to 1.0 for every method, including
  Source-only. Without it, DANN and CDAN diverge within the first epoch (the gradient-reversal
  layer maximises an unbounded domain loss, so features and discriminator weights grow without
  limit). The pre-clipping norm is logged as `grad_norm` in `history.json`.
- **Batches and epochs:**
  - One epoch is ⌊source-train images / 24⌋ updates.
  - Each update uses 8 + 8 + 8 source images and, for adaptation methods, 24 target images.
- **Domain-separability probe:**
  - Features are standardised, with the scaler fitted on the probe's training part only.
  - Logistic regression with `class_weight="balanced"`, C = 1.
  - A seed-6304 stratified 70/30 split.
  - Sketch features are subsampled to the number of source-validation features.

## Attribution

- **PACS:** Li et al., "Deeper, Broader and Artier Domain Generalization", ICCV 2017. Download link from [DomainBed](https://github.com/facebookresearch/DomainBed).
- **Methods:** DANN (Ganin et al., 2016), CDAN (Long et al., 2018) and DAN/MKMMD (Long et al., 2015), implemented from the papers' formulas.
