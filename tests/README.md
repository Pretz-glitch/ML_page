# `tests/` — unit tests

Run with:

```bash
pytest -q
```

## Planned tests

| Test file | What it asserts |
|---|---|
| `test_split.py` | Train and test day ranges are disjoint and match 0–422 / 423–704; scaler statistics change if test data is included |
| `test_scaling.py` | Per-station medians are approximately zero after scaling; IQR is approximately one; no test statistics used |
| `test_faults.py` | Injected rows are labelled; clean rows are unchanged; prevalence matches; values stay in [0, 1]; training data untouched |
| `test_detectors.py` | Score orientation is correct (a known outlier scores higher); covariance is positive definite or uses a pseudo-inverse |
| `test_metrics.py` | AUC is 1.0 for perfect separation and 0.5 for random scores; top-1 recovery is 1.0 for a perfect attribution |
| `test_explain.py` | Mahalanobis contributions sum to the squared distance; occlusion returns one contribution per feature |

## Rules

- Tests must not require network access.
- Tests must not depend on generated result files.
- Keep tests small: use a synthetic 5×3 array where possible.
- A bug found in the pipeline gets a regression test before it is fixed.
