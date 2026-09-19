# Task 1 — Inductive Biases and Feature Representations

Compares a frozen ResNet-50, ViT-B/16 and CLIP ViT-B/32 (linear heads + CLIP zero-shot)
on STL-10 under controlled interventions: grayscale, hue rotation, AdaIN shape/texture
cue conflicts, translation and 4x4 patch shuffling. Also measures how much each
intervention moves the frozen representation.

## 0. One-time setup (from the repository root)

```bash
pip install -r requirements.txt

# AdaIN weights (Huang & Belongie 2017, released by naoto0804/pytorch-AdaIN)
mkdir -p third_party/adain
wget -O third_party/adain/vgg_normalised.pth https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth
wget -O third_party/adain/decoder.pth        https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth
```

STL-10 (~2.6 GB) downloads automatically into `data/stl10/` on first use.
A GPU is strongly recommended (a Colab T4 is enough).

## 1. Before running anything: pre-register your choices

Open `configs/task1.yaml`. Every `[DESIGN CHOICE]` line is yours to decide.
For each one, write a hypothesis and the metric that will test it into your report notes,
**before** looking at any result:

| Choice | Default here | Metric that tests it |
|---|---|---|
| Dataset | STL-10 | — |
| Extra colour intervention | 180° hue rotation | Δaccuracy and consistency vs clean |
| Cue-conflict pairs and style strength | 5 pairs covering all 10 classes, alpha = 1.0 | shape bias and coverage |
| Rejection rule thresholds | edge corr ≥ 0.25, style gain ≥ 0.50, pixel std ≥ 0.05 | accepted/rejected counts |
| Visualisation | t-SNE, perplexity 30, cosine metric, PCA init | qualitative (settings saved) |

## 2. Run, in this order

```bash
python -m task1.data.make_subset        --config task1/configs/task1.yaml   # splits + 500-image subset
python -m task1.data.make_cue_conflicts --config task1/configs/task1.yaml   # AdaIN images + review sheet
#   -> look at task1/results/cue_conflicts/sheets/*.png
#   -> in task1/results/cue_conflicts/review.csv, type 1 or 0 in `manual_accept`
#      wherever the automatic decision is visibly wrong (leave blank to keep the auto decision)
#   -> do this BEFORE the next command: model predictions must never influence retention
python -m task1.scripts.run_task1       --config task1/configs/task1.yaml   # everything else
```

Features are cached in `task1/results/cache/`. Add `--recompute` to re-extract them.
If fewer than 200 conflicts survive, increase `cue_conflict.per_direction`, delete
`review.csv`, and regenerate. Do not loosen thresholds after seeing model outputs.

## 3. Where each piece of "Required Evidence" ends up (`task1/results/`)

| Handout requirement | File(s) |
|---|---|
| Clean / grayscale / extra-colour / patch-shuffle comparison | `table_clean_color_shuffle.csv` (all conditions: `table_all_conditions.csv`) |
| Shape / texture / other counts, shape bias, coverage | `table_shape_bias.csv`, `table_shape_bias_per_pair.csv`, `cue_conflict_counts.json`, `cue_conflicts/generation_summary.json` |
| Translation curve | `figures/translation_curve.png`, `table_translation.csv` |
| Representation stability | `table_representation_stability.csv` (includes a random-pair baseline and cosine for flipped vs unchanged predictions) |
| t-SNE of clean vs transformed | `figures/tsne_per_backbone.png` (required), `tsne_settings.json`, `figures/tsne_supplementary_grid.png` (optional) |
| Informative cue-conflict cases | `figures/cue_conflict_examples.png`, `cue_conflict_decisions.csv` |
| Extras for RQ3 | `table_clip_zeroshot_vs_head_agreement.csv`, `linear_heads.json` |
| Reproducibility | `data/splits/*.json` (split and subset ids), `patch_permutations.json`, `cue_conflicts/review.csv` |

## 4. File map

```
task1/
  configs/task1.yaml          every number that affects a result
  data/datasets.py            STL-10 loading + the shared 224x224 [0,1] canvas
  data/make_subset.py         80/20 split, 500-image balanced test subset (seed 6304)
  data/transforms.py          grayscale, hue rotation, translation, patch shuffle
  data/adain.py               AdaIN networks (verbatim architecture from naoto0804/pytorch-AdaIN)
  data/make_cue_conflicts.py  generation + pre-registered, model-free rejection rule
  models/backbones.py         frozen ResNet-50 / ViT-B/16 / CLIP wrappers (+ zero-shot)
  models/linear_head.py       linear probe (AdamW 1e-3, wd 1e-4, <=50 epochs, patience 5)
  analysis/evaluate_bias.py   accuracy, macro-F1, confidence, consistency, shape bias
  analysis/feature_similarity.py  cosine stability I_T and its baselines
  analysis/representation.py  t-SNE + plotting
  scripts/run_task1.py        driver that produces every table and figure
```

## Attribution

`data/adain.py` reproduces the VGG encoder and decoder architecture of
[naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) (MIT licence) so
that its released weights load unchanged. Backbone weights come from torchvision and
[OpenCLIP](https://github.com/mlfoundations/open_clip).
