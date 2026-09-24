# Run 1 (recovered)

Run 1 was the first full Task 4 run (interactive Kaggle session, 20 Sep 2026, code version `task4-v1`). It
finished without errors, but the session ended before its output zips were downloaded. Its executed notebook is
kept as `notebooks/task4_kaggle_run1_executed.ipynb`.

`run1_key_numbers.json` holds the numbers recovered from that notebook's printed output:
- the checkpoint hash prefixes, epochs and validation accuracies;
- the split hash;
- the freeze time;
- Tables 1 and 2 (2 decimals);
- the per-epoch training logs of all three models.

Run 2, the committed Kaggle run whose files are in `task4/results/`, prints a side-by-side comparison against
these numbers. The reported results come from run 2; run 1 serves as a reproducibility check. Between the two runs
no file that affects the three pre-registered models, their scores or their thresholds changed. `train.py` only
additionally writes `last.pt`, which changes neither training nor `best.pt`.
