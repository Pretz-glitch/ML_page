# `src/` — pipeline modules

Implement in this order. Each module is importable and testable without running a full experiment.

| Module | Status | Responsibility |
|---|---|---|
| `io.py` | ✅ Phase 1 | Load `water_dataset.mat`, build the `(705, 37, 11)` tensor, station IDs, groups, feature names and dates |
| `preprocess.py` | ✅ Phase 1 | Flatten to a station-day table, apply the temporal split, fit within-station median/IQR scaling on train only |
| `audit.py` | ✅ Phase 1 | Variance decomposition, correlations, floor checks, station/group summaries |
| `plots.py` | ✅ Phase 1 | Audit figures (correlation, variance, coverage, distributions) |
| `utils.py` | ✅ Phase 1 | Project paths, git commit, package versions, checksums, JSON logs |
| `faults.py` | ⬜ Phase 2 | Spike, drift, correlation-break and stuck injectors; return corrupted data plus a tidy label table |
| `detectors.py` | ⬜ Phase 2 | MAD, Mahalanobis, Isolation Forest, optional Extended IF |
| `explain.py` | ⬜ Phase 4 | Occlusion attribution, exact Mahalanobis contribution, optional DIFFI |
| `metrics.py` | ⬜ Phase 3–4 | AUC-ROC/PR, recall@k, precision@k, FPR, top-1 recovery, MRR, deletion drop, bootstrap CIs |

## Conventions

- Every public function has a docstring and type hints.
- No function reads from or writes to disk except `io.py` and the scripts.
- Seeds are explicit arguments, never global state.
- Scaled data is the modelling representation; raw data is kept for case-study plots.
- `Y_tr` / `Y_te` are never loaded into the modelling path.
- Imports from `src` require the project root on `sys.path`; the scripts and `tests/conftest.py` handle this.
