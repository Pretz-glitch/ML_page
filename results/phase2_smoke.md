# Phase 2 smoke test

**Status:** end-to-end integration test. Not the final Phase 3 benchmark.

Regenerate with `python scripts/phase2_smoke.py`.

## Setup

| Item | Value |
|---|---|
| Data | 26,085 station-days; train days 0–422, test days 423–704 |
| Scaling | Within-station median/IQR, fitted on train only |
| Injections | Spike, drift, correlation break, stuck |
| Prevalence | 2% of test station-days |
| Magnitude | 4–8 × training IQR (spike, drift) |
| Windows | 5–20 days (drift, correlation break, stuck) |
| Seed | 0 |
| Eligible features | `pH_max`, `SC_mean`, `DO_max`, `DO_mean`, `DO_min`, `T_mean`, `T_min`, `T_max` |

## Detection (ROC-AUC)

| Detector | spike | drift | correlation_break | stuck |
|---|---:|---:|---:|---:|
| mad | 0.805 | 0.763 | 0.532 | 0.576 |
| mahalanobis | 0.994 | 0.979 | 0.863 | 0.794 |
| pca | 0.890 | 0.872 | 0.658 | 0.567 |
| iforest | 0.748 | 0.759 | 0.568 | 0.579 |
| eif | 0.930 | 0.905 | 0.645 | 0.592 |

## Recall at k (k = number injected)

| Detector | spike | drift | correlation_break | stuck |
|---|---:|---:|---:|---:|
| mad | 0.086 | 0.009 | 0.009 | 0.023 |
| mahalanobis | 0.756 | 0.724 | 0.379 | 0.192 |
| pca | 0.167 | 0.076 | 0.033 | 0.009 |
| iforest | 0.014 | 0.058 | 0.009 | 0.056 |
| eif | 0.215 | 0.178 | 0.019 | 0.028 |

## Integrity checks

- Clean cells are byte-identical to the input tensor.
- Every marked cell changes value.
- All corrupted values remain in [0, 1].
- Cell mask count equals the number of label rows.
- Station-day mask count equals the number of unique labelled station-days.
- Label `row_id` matches `day_index × 37 + station_index`.
- Every label maps to a marked cell.
- Dates, windows and magnitudes are internally consistent.
- Injection is deterministic under the same seed.

## What this shows

- The covariance detector recovers the pilot result (spike 0.994, drift 0.979, correlation break 0.863).
- Standard Isolation Forest is weaker on every fault type, especially correlation breaks (0.568).
- Extended Isolation Forest recovers much of the isolation-model gap (0.930 / 0.905 / 0.645).
- Stuck faults are weak for most cross-sectional detectors (0.57–0.59). Mahalanobis reaches 0.79 because a frozen value can still violate the conditional expectation given the other features, but the signal is much weaker than for spike or drift.

The final Phase 3 run will use 5 seeds, the full detector × fault matrix, and 95% confidence intervals.
