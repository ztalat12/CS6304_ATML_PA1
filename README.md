# CS6304 / EE-5102 ATML — PA1: Learning Beyond IID

Four studies of what happens when the i.i.d. assumption fails: inductive biases
under appearance interventions (Task 1), unsupervised domain adaptation
(Task 2), domain generalization (Task 3), and open-set recognition (Task 4).

**Seed 6304 everywhere.** Every number in the report traces to a file in
`taskN/results/`. The report PDF is in `report/`.

---

## Headline results

| Task | Setting | Headline |
|---|---|---|
| 1 | STL-10, three frozen backbones | Clean accuracy spans 96.4–97.4 while shape bias spans 49.6–85.4% |
| 2 | PACS → Sketch, unlabelled target allowed | All adaptation underperforms Source-only (69.59%): DAN −11.00, CDAN −11.10, DANN −43.14 |
| 3 | PACS → Sketch, no target access | Nothing distinguishable from ERM; SAM cuts sharpness 62% with no transfer gain |
| 4 | CIFAR-10 known / CIFAR-100 unknown | Near unknowns cost ~10 AUROC points vs far; vanilla MSP 80.96 near / 90.25 far |

---

## Repository layout

```
.
├── common/                  # shared utilities (seed, io, metrics)
├── shared/                  # PACS protocol + split indices for Tasks 2 and 3
│   └── splits/              # pacs_sources_seed6304.json, pacs_sketch_target.json
├── task1/                   # inductive biases and representations
├── task2/                   # unsupervised domain adaptation
├── task3/                   # domain generalization
├── task4/                   # open-set recognition
├── notebooks/               # Colab / Kaggle driver notebooks
├── report/                  # PA1_report.tex, references.bib, figures/, PDF
├── requirements.txt
└── .gitignore
```

Each `taskN/` holds `configs/`, method and data modules, `results/`, and its own
`README.md` with the exact commands for that task.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tasks 1 and 4 were run on a single T4/P100 GPU (Colab and Kaggle respectively);
Tasks 2 and 3 on Kaggle. CPU will work but Task 4 takes roughly 2.5 hours on a
P100 and considerably longer without a GPU.

## Data

Raw datasets are **not** committed. Split indices are.

| Dataset | Used by | How to obtain |
|---|---|---|
| STL-10 | Task 1 | Downloaded automatically by torchvision into `data/stl10` on first run (~2.6 GB) |
| PACS | Tasks 2, 3 | Download separately and point `data_root` at the folder containing `photo/`, `art_painting/`, `cartoon/`, `sketch/` |
| CIFAR-10, CIFAR-100 | Task 4 | Downloaded automatically by torchvision into the configured `data_root` |
| AdaIN VGG + decoder weights | Task 1 | From the `naoto0804/pytorch-AdaIN` release; place at `third_party/adain/` (see `task1/README.md`) |

All split indices are committed under `taskN/data/splits/` and
`shared/splits/`, so splits are reproducible without re-deriving them.

---

## Reproducing each task

Run from the repository root. Each task's README has the full detail; these are
the minimum paths.

### Task 1 — inductive biases

```bash
python -m task1.data.make_subset        --config task1/configs/task1.yaml
python -m task1.data.make_cue_conflicts --config task1/configs/task1.yaml
# manual review step: fill manual_accept in task1/results/cue_conflicts/review.csv
python -m task1.scripts.run_task1       --config task1/configs/task1.yaml
```

The manual review of the generated cue-conflict images happens **before** any
model is evaluated; it is part of the protocol, not a post-hoc filter.

### Task 2 — unsupervised domain adaptation

```bash
for CFG in source_only dan dann cdan; do
  python -m task2.train --config task2/configs/$CFG.yaml --set data_root=<PACS>
done
python -m task2.evaluate_final --set data_root=<PACS>
python -m task2.evaluate_final --set data_root=<PACS> --study dan
python -m task2.evaluate_final --set data_root=<PACS> --study dann
```

`evaluate_final` is the only script that reads Sketch labels, and it writes a
frozen record of source-validation scores before doing so.

### Task 3 — domain generalization

Task 3 **reuses the Task 2 Source-only checkpoint as its ERM baseline** and
refuses to start if the protocol does not match it.

```bash
for CFG in dan_dg sam; do
  python -m task3.train --config task3/configs/$CFG.yaml --set data_root=<PACS>
done
python -m task3.evaluate_sources --set data_root=<PACS> \
       erm_checkpoint=task2/results/runs/source_only/best.pt
python -m task3.evaluate_sketch  --set data_root=<PACS> \
       erm_checkpoint=task2/results/runs/source_only/best.pt
python -m task3.evaluate_sketch  --study dan_dg --set data_root=<PACS> \
       erm_checkpoint=task2/results/runs/source_only/best.pt
```

### Task 4 — open-set recognition

Order matters: PROSER initialises from the selected Vanilla checkpoint, and no
CIFAR-100 image may be touched until the freeze record is written.

```bash
python -m task4.train --config task4/configs/vanilla.yaml --set data_root=<CIFAR>
python -m task4.train --config task4/configs/gcsc.yaml    --set data_root=<CIFAR>
python -m task4.train --config task4/configs/proser.yaml  --set data_root=<CIFAR>
python -m task4.extract_outputs --set data_root=<CIFAR>     # writes the freeze record
python -m task4.evaluate_osr    --set data_root=<CIFAR>     # first CIFAR-100 access
python -m task4.evaluation.failure_analysis
python -m task4.evaluation.apply_plausibility_map           # optional, judgement only
python -m task4.posthoc.bn_diagnosis --set data_root=<CIFAR>
python -m task4.posthoc.evaluate_variant --set data_root=<CIFAR>
```

Add `--help` to any script for its full flag list.

---

## Where results live

| Path | Contents |
|---|---|
| `task1/results/*.csv` | accuracy, shape bias, translation, stability, agreement tables |
| `task1/results/figures/` | translation curve, t-SNE, cue-conflict examples |
| `task2/results/` | main and study tables, training curves, per-class analysis, frozen records |
| `task3/results/final/` | main and study tables, collapse diagnostics, protocol.json, frozen records |
| `task4/results/final/` | Table 1 and 2, AUROC/McNemar statistics, failure analysis, figures |
| `task4/results/frozen/` | freeze record, Mahalanobis statistics, unknown-access log |
| `task4/results/posthoc/` | BatchNorm diagnosis and the corrected PROSER variant |

Model checkpoints (`*.pt`) and feature caches are git-ignored; every number in
the report is recoverable from the committed CSV and JSON files without them.

---

## Reproducibility and integrity

- **Seed 6304** fixes weight initialisation, data shuffling and augmentation in
  every task.
- **Task 4 was executed twice and reproduced bit-identically**: all three
  checkpoint SHA-256 digests matched (`da540bbd…`, `f064d95b…`, `27526c62…`),
  as did the split hash and every cell of Tables 1 and 2.
- **Task 2 leakage control.** Checkpoints were selected on mean
  source-validation macro-F1 only. Frozen records (`task2_frozen_*.json`) were
  written at 08:28:37, 08:29:58 and 08:31:00 on 20 Sep 2026, before Sketch was
  evaluated.
- **Task 3 leakage control.** No Sketch image reached training, diagnostics or
  selection; frozen records at 08:55:56 and 08:56:47 the same day.
  `task3/results/final/protocol.json` records `erm_checkpoint` pointing at the
  Task 2 run, confirming ERM was loaded rather than retrained.
- **Task 4 leakage control.** The freeze record (twelve code files hashed, plus
  checkpoints, score definitions and thresholds) was written at
  `2026-09-21T07:35:54Z`; the first CIFAR-100 access was logged at
  `08:27:39Z`, 51 minutes later. The post-hoc BatchNorm variant has its own
  later record at `08:56:38Z` pointing back at the pre-registered one.
- **One seed per configuration.** Significance tests address variation over
  images, not over training runs. See the report's Limitations.

---

## Attributions

- AdaIN style transfer for cue conflicts: Huang & Belongie (2017); VGG encoder
  and decoder weights from the `naoto0804/pytorch-AdaIN` release.
- PROSER classifier- and data-placeholder losses follow Zhou et al. (2021) and
  the authors' reference implementation.
- Backbones are torchvision and OpenCLIP pretrained weights as specified in
  each task's config.

## Report

`report/PA1_report.tex` (+ `references.bib`, `figures/`). Compile with
`pdflatex → bibtex → pdflatex → pdflatex`.
