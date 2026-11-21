# `src/` — pipeline modules

Implement in this order. Each module is importable and testable without running a full experiment.

| Module | Status | Responsibility |
|---|---|---|
| `io.py` | ✅ Phase 1 | Load `water_dataset.mat`, build the `(705, 37, 11)` tensor, station IDs, groups, feature names and dates |
| `preprocess.py` | ✅ Phase 1 | Flatten to a station-day table, apply the temporal split, fit within-station median/IQR scaling on train only |
| `audit.py` | ✅ Phase 1 | Variance decomposition, correlations, floor checks, station/group summaries |
| `plots.py` | ✅ Phase 1 | Audit figures (correlation, variance, coverage, distributions) |
| `utils.py` | ✅ Phase 1 | Project paths, git commit, package versions, checksums, JSON logs |
| `detectors.py` | ✅ Phase 2 | MAD, Mahalanobis, PCA reconstruction, Isolation Forest, Extended IF |
| `faults.py` | ✅ Phase 2 | Spike, drift, correlation-break and stuck injectors; corrupted tensor + labels + masks |
| `explain.py` | ⬜ Phase 4 | Occlusion attribution, exact Mahalanobis contribution, optional DIFFI |
| `metrics.py` | ✅ Phase 3 | AUC-ROC/PR, recall@k, precision@k, FPR, bootstrap CIs, paired bootstrap differences |

## Conventions

- Every public function has a docstring and type hints.
- No function reads from or writes to disk except `io.py` and the scripts.
- Seeds are explicit arguments, never global state.
- Scaled data is the modelling representation; raw data is kept for case-study plots.
- `Y_tr` / `Y_te` are never loaded into the modelling path.
- Imports from `src` require the project root on `sys.path`; the scripts and `tests/conftest.py` handle this.

## Detector defaults

- **Mahalanobis** uses the sample covariance with a pseudo-inverse. This preserves the dependence structure that correlation-break faults violate. Ledoit–Wolf shrinkage is available as `shrinkage=True` but is **not** the default: on this dataset it shrinks the off-diagonals heavily and weakens dependency-anomaly detection.
- **Isolation Forest** uses 300 trees, `max_samples="auto"`, `contamination="auto"`.
- **Extended Isolation Forest** uses `isotree` with `ndim=n_features` (hyperplane splits).
- **PCA reconstruction** uses three components by default.

## Fault-injection conventions

- Injection happens on **raw normalised values in [0, 1]**, before scaling.
- The scaler is fitted on clean training data only and never sees corrupted values.
- Every injection returns a tidy label table with `event_id`, `row_id`, `day_index`, `date`, `station_id`, `station_index`, `feature`, `feature_index`, `fault_type`, `magnitude`, `window_start_day`, `window_end_day` and `seed`.
- Two boolean masks are returned: `cell_mask` and `station_day_mask`.
- Features are eligible only if their median within-station IQR exceeds 0.01 in the raw normalised units.
