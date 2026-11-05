# `tests/` — unit tests

Run with:

```bash
pytest -q
```

## Current tests (Phase 1, 16 passing)

| Test file | What it asserts |
|---|---|
| `conftest.py` | Loads the raw archive once for the session |
| `test_io.py` | Tensor shape `(705, 37, 11)`, 26,085 station-days, finite values, contiguous dates, 37 unique stations, 11 expected feature names, complete station-group coverage, values in `[0, 1]` |
| `test_split.py` | 26,085 rows, 423/282 day split, no overlap, scaler fitted on training medians only, training medians are zero after scaling, training IQR is one where IQR was used, test rows transformed with train parameters |

## Planned tests

| Test file | What it asserts |
|---|---|
| `test_faults.py` | Injected rows are labelled; clean rows are unchanged; prevalence matches; values stay in `[0, 1]`; training data untouched |
| `test_detectors.py` | Score orientation is correct; covariance is positive definite or uses a pseudo-inverse |
| `test_metrics.py` | AUC is 1.0 for perfect separation and 0.5 for random scores; top-1 recovery is 1.0 for a perfect attribution |
| `test_explain.py` | Mahalanobis contributions sum to the squared distance; occlusion returns one contribution per feature |

## Rules

- Tests must not require network access.
- Tests must not depend on generated result files.
- Keep tests small: use a synthetic 5×3 array where possible.
- A bug found in the pipeline gets a regression test before it is fixed.
