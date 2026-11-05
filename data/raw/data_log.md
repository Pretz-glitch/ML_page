# Data log

## Download record

| Field | Value |
|---|---|
| Dataset | UCI Machine Learning Repository, *Water Quality Prediction* (733) |
| URL | https://archive.ics.uci.edu/static/public/733/water+quality+prediction-1.zip |
| Downloaded | 2026-11-05 |
| Archive | `water+quality+prediction-1.zip` (not committed) |
| Extracted | `water_dataset.mat`, `README.docx` |
| Licence | CC BY 4.0 |

## Checksums (SHA-256)

| File | SHA-256 |
|---|---|
| `water_dataset.mat` | `228fd024d0c288247c5bbdf5eaa12404d69d0ec915003802c30908b3d2648ffd` |
| `README.docx` | `0597a57e04c572a01d4f1768ffd093cad668e489f2056f44e66bd003e94d129e` |

## Verified structure

```python
import scipy.io as sio
m = sio.loadmat("water_dataset.mat")

m["X_tr"].shape   # (1, 423) object; each cell (37, 11)
m["X_te"].shape   # (1, 282) object; each cell (37, 11)
m["Y_tr"].shape   # (37, 423)
m["Y_te"].shape   # (37, 282)
m["features"].shape    # (1, 11)
m["location_ids"].shape  # (37, 1)
m["location_group"].shape  # (1, 3)
```

## Notes

- The README says "36 sites"; the arrays contain **37**. Use 37 and document the discrepancy.
- The zip also contains a `__MACOSX/` folder; ignore it.
- No missing values are present, but many values sit near the floor of the normalisation (≈0.001). Inspect before modelling.
- All 11 features are normalised, not only pH.
- `Y_tr` / `Y_te` are median pH and are **not used** in this project.

## Citation

Zhao, L., Gkountouna, O., & Pfoser, D. (2019). Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints. *ACM Transactions on Spatial Algorithms and Systems*, 5(3), Article 19. DOI: 10.1145/3339823
