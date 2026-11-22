# `tests/` — unit tests

Run with:

```bash
pytest -q
```

## Current tests (Phases 1–5, 88 passing)

| Test file | What it asserts |
|---|---|
| `conftest.py` | Loads the raw archive once for the session |
| `test_io.py` | Tensor shape `(705, 37, 11)`, 26,085 station-days, finite values, contiguous dates, 37 unique stations, 11 expected feature names, complete station-group coverage, values in `[0, 1]` |
| `test_split.py` | 26,085 rows, 423/282 day split, no overlap, scaler fitted on training medians only, training medians are zero after scaling, training IQR is one where IQR was used, test rows transformed with train parameters |
| `test_detectors.py` | Every detector flags a known dependency-breaking outlier above its 99th percentile; score shape and finiteness; width checks; covariance symmetry; factory errors |
| `test_faults.py` | Injected rows are labelled; clean cells are byte-identical; marked cells change; values stay in `[0, 1]`; prevalence matches; labels are unique; row IDs match the table convention; masks agree with labels; stuck values are constant per event; drift magnitude is constant per event; same seed is deterministic; invalid inputs raise |
| `test_metrics.py` | Perfect separation gives AUC 1.0; known recall@k and FPR; invalid inputs raise; bootstrap CIs are reproducible and ordered; paired bootstrap detects a positive difference and handles zero difference |
| `test_explain.py` | Occlusion and conditional-occlusion shapes; conditional means match the Gaussian formula; Mahalanobis contributions sum to the squared distance; ranking, true rank, deletion metrics and majority agreement |
| `test_realdata.py` | Pairwise overlap, concentration summaries, flag summaries, maximum score ratio, and case selection with station/feature diversity |

## Planned tests

Phase 6 will add an integration check that the README reproduction path and the report tables agree. No other unit tests are planned.

## Rules

- Tests must not require network access.
- Tests must not depend on generated result files.
- Keep tests small: use synthetic arrays where possible.
- A bug found in the pipeline gets a regression test before it is fixed.
