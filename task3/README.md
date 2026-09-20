# Task 3 — Domain Generalization on PACS

**Setup.** Same PACS protocol, splits, model, optimiser, budget and seed as Task 2. The difference:
**no Sketch image may be touched** by training, source-side diagnostics, checkpoint selection or
hyperparameter selection. Sketch is loaded by exactly one script, `task3/evaluate_sketch.py`.

**Methods compared**
1. **ERM** — the Task 2 Source-only checkpoint, loaded, never retrained.
2. **DAN-DG** — ERM loss + (λ_DG/3)·Σ_{e<e'} MMD²(F(X_e), F(X_e')) over the three source-domain pairs,
   using the same MMD code and kernels as Task 2's DAN. The only difference from DAN is *which*
   distributions are aligned, which is what makes the Task 2 vs Task 3 comparison meaningful.
3. **SAM** — non-adaptive sharpness-aware minimisation, ρ = 0.05, two forward/backward passes per
   update, frozen-BatchNorm policy in both.

## 1. Train (ERM is not retrained)

```bash
python -m task3.train --config task3/configs/dan_dg.yaml
python -m task3.train --config task3/configs/sam.yaml
# controlled study -- ONE family, fixed before any Sketch evaluation (default: DAN-DG lambda,
# so it can be compared directly with Task 2's DAN lambda study)
python -m task3.train --config task3/configs/study_dan_dg_lambda0.1.yaml
python -m task3.train --config task3/configs/study_dan_dg_lambda10.yaml
#   (alternative: study_sam_rho0.01.yaml and study_sam_rho0.1.yaml)
```

`task3/train.py` refuses to start if the Task 3 config differs from the protocol stored in the ERM
checkpoint's `config_resolved.json` (seed, workers, batch size, lr, weight decay, gradient clipping,
epochs, patience). If you change `task2/configs/base.yaml` after training Source-only, retrain
Source-only too, otherwise ERM and the Task 3 methods are not comparable.

## 2. Evaluate, in two steps

```bash
python -m task3.evaluate_sources                 # Step A: SOURCE ONLY, no Sketch is loaded
python -m task3.evaluate_sketch                  # Step B: the only Sketch access
python -m task3.evaluate_sources --study dan_dg  # (then the study, same two steps)
python -m task3.evaluate_sketch  --study dan_dg
```

Step A produces the per-domain, mean and worst source results, the source-domain separability score
and the sharpness proxy. Run it, look at it, and only then run Step B: that ordering is what makes
"settings were frozen before Sketch was seen" verifiable rather than a claim.

## 3. Where each piece of required evidence ends up (`task3/results/final/`)

| Handout requirement | File(s) |
|---|---|
| ERM / DAN-DG / SAM per source domain, mean, worst, Sketch acc + macro-F1, Δ vs ERM | `table_main.csv` (source half: `source_side.csv`) |
| Source-domain separability (chance 33.3%) and the shared sharpness proxy | `source_side.csv`, columns `src_domain_sep`, `sharpness` |
| Training curves incl. the MMD penalty | `figures/training_curves.png`, `runs/*/history.json` |
| Controlled study | `source_side_study_<name>.csv`, `table_study_<name>.csv` |
| Per-class Sketch changes, failures, comparison with Task 2 | `per_class_sketch_acc.csv`, `class_analysis.json`, `figures/per_class_delta.png`, `figures/failures_*.png`, `task2_vs_task3_per_class.csv`, `task2_vs_task3_overall.json` |
| Audit record | `protocol.json`, `frozen_main.json` |

## 4. File map

```
task3/configs/          base.yaml inherits task2/configs/base.yaml, so the protocol cannot drift
task3/methods/erm.py    resolves the Task 2 checkpoint (nothing to train)
task3/methods/dan_dg.py pairwise source MMD
task3/methods/sam.py    two-pass SAM update
task3/evaluation/source_diagnostics.py   source-domain separability + the shared sharpness proxy
task3/train.py          trains one config; never imports the Sketch loader
task3/evaluate_sources.py   Step A (source only)
task3/evaluate_sketch.py    Step B (Sketch)
task3/diagnose_collapse.py  source-only diagnosis of the DAN-DG collapse (optional)
task3/configs/diag_dan_dg_unbiased.yaml  optional diagnostic run (unbiased MMD, lambda_DG = 1)
```

## Implementation choices to state in the report

- **DAN-DG pairs:** each update has 8 images per source domain, so every pair's MMD is estimated from
  8 + 8 features, with the median bandwidth computed per pair. Small-sample MMD estimates are noisy;
  this follows from the handout's batch composition.
- **MMD estimator:** the same biased (V-statistic) estimate as Task 2. *(Corrected: an earlier
  version of this README said the difference from the unbiased estimate was a constant with no
  effect on gradients. That is wrong.)* The biased estimate equals the unbiased one plus
  6/m − (1/m)·(mean within-domain kernel similarity). The constant part means the logged `loss_mmd`
  cannot go below ≈ 0.47 for 8-vs-8 pairs (≈ 0.15 for Task 2's 24 vs 24) even when two domains
  are identical. The second part has a gradient. Because the median bandwidth is detached, that
  gradient pulls every feature towards a smaller norm in every batch, and the pull is three times
  stronger at 8 vs 8 than at 24 vs 24. See `shared/mmd.py`.
- **SAM:** the perturbation uses the gradient of the *same* augmented batch in both passes; ε is
  removed before the AdamW step, so the update is applied at θ with the perturbed gradient.
  `loss_sam` (loss at θ+ε) is logged next to `loss_cls` (loss at θ).
- **Sharpness proxy:** one ascent step of radius 0.05 on a fixed 96-image validation batch
  (32 per source, seed 6304), model in eval mode, all trainable parameters perturbed including BN
  γ and β. It is a local, parameterisation-dependent measure, not global flatness.
- **Gradient clipping** (if enabled in `task2/configs/base.yaml`) applies here too, including inside
  SAM's second pass. Whatever value was used for the ERM checkpoint must be used here.

## DAN-DG at λ_DG = 1 collapsed: what it means and how it is handled

**What happened (source data only).** From epoch 1, `loss_cls` stayed at 1.92–1.94. The best
constant predictor, which just outputs the class frequencies, scores 1.913. Mean source-val
macro-F1 was 5.07, exactly the score for predicting "person" for every image. The run
early-stopped at epoch 6. Task 2's DAN at λ = 10 shows the identical fingerprint.

**Most likely cause.** The biased estimator pulls features to shrink (see above). Its strength is
λ times the 8-vs-8 or 24-vs-24 offset (about 0.45 and 0.15):

| run | λ × pull | outcome |
|---|---|---|
| Task 2 DAN λ = 0.1 / 1 | 0.015 / 0.15 | trained (source F1 93.9 / 95.0) |
| Task 3 DAN-DG λ = 0.1 | 0.045 | trained (F1 92 after 2 epochs) |
| Task 3 DAN-DG λ = 1 | 0.45 | collapsed |
| Task 2 DAN λ = 10 | 1.5 | collapsed |
| Task 3 DAN-DG λ = 10 | 4.5 | expected to collapse |

The collapsed state is not a better optimum of the λ = 1 objective. Its training loss is 2.40,
while the λ = 0.1 model already scores ≈ 0.74 on the same objective after two epochs. So this is
an optimisation failure, not the method "choosing" invariance over accuracy.

**How it is handled.**
1. The main comparison keeps λ_DG = 1 with the biased estimator, as the handout requires. The
   collapse is reported as a result: it is exactly the "alignment removed class-discriminative
   information" case that RQ2 asks about.
2. `python -m task3.diagnose_collapse` (source only) shows where the collapse happened. It reports
   the most-predicted class, feature norms, a 7-way class probe on frozen features, and the
   8-vs-8 MMD with and without the estimator offset.
3. Optional, only with the TA's agreement: `task3/configs/diag_dan_dg_unbiased.yaml` repeats
   λ_DG = 1 with the unbiased estimator. It is a labelled diagnostic, never the main row, and is
   not evaluated on Sketch.
4. `notebooks/rerun_task2_task3_kaggle.ipynb` re-trains every Task 2 and Task 3 run from scratch and
   compares each with its original run, to show whether any result changes from run to run. It also
   repeats the two MMD collapses with seed 6305 (`--set seed=6305 run_name=... seed_check=true`;
   `seed_check` lets `task3/train.py` accept a seed that differs from the ERM checkpoint's). That is a
   robustness check only, never part of a comparison.

## Attribution

SAM follows Foret et al. (2021); the MMD mechanism follows Long et al. (2015). Both are implemented
from the papers' formulas, reusing the Task 2 MMD code unchanged.
