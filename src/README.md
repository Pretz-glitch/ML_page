# `src/` — pipeline modules

Planned modules. Implement in this order; each should be importable and testable without running a full experiment.

| Module | Responsibility |
|---|---|
| `io.py` | Load `water_dataset.mat`, build the `(705, 37, 11)` tensor, expose feature names and station metadata |
| `preprocess.py` | Flatten to a station-day table, apply the temporal split, fit within-station median/IQR scaling on train only |
| `faults.py` | Spike, drift, correlation-break and stuck injectors; return corrupted data plus a tidy label table |
| `detectors.py` | MAD, Mahalanobis (Ledoit–Wolf or pseudo-inverse), Isolation Forest, optional Extended IF |
| `explain.py` | Occlusion attribution, exact Mahalanobis contribution, optional DIFFI |
| `metrics.py` | AUC-ROC/PR, recall@k, precision@k, FPR, top-1 recovery, MRR, deletion drop, bootstrap CIs |
| `plots.py` | Shared figure styling and helpers |

## Conventions

- Every public function has a docstring and type hints.
- No function reads from or writes to disk except the `io` module and the scripts.
- Seeds are explicit arguments, never global state.
- Scaled data is the modelling representation; raw data is kept for case-study plots.
- `Y_tr` / `Y_te` are never loaded into the modelling path.

## Not yet implemented

The modules are placeholders until Milestone 1. Do not add empty files with no behaviour; implement each module with its tests when the milestone starts.
