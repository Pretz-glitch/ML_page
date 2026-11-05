# Data

## Source

- **Dataset:** UCI Machine Learning Repository, *Water Quality Prediction* (dataset 733)
- **URL:** https://archive.ics.uci.edu/dataset/733/water+quality+prediction-1
- **Licence:** CC BY 4.0
- **Original paper:** Zhao, L., Gkountouna, O., & Pfoser, D. (2019). *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints.* ACM Transactions on Spatial Algorithms and Systems, 5(3), Article 19. DOI: 10.1145/3339823
- **Upstream source:** United States Geological Survey (USGS)

## Files

| File | Size | Description |
|---|---|---|
| `raw/water_dataset.mat` | ~1.0 MB | Original MATLAB archive |
| `raw/README.docx` | ~16 KB | Original dataset README |
| `raw/data_log.md` | — | Download date, checksum, structure notes |
| `processed/station_day.csv` | generated | 26,085-row station-day table (gitignored) |

## Structure

```
X_tr:  1×423 cell   each cell is 37×11   (37 stations × 11 features per day)
X_te:  1×282 cell   each cell is 37×11
Y_tr:  37×423       median pH, normalised   ← not used in this project
Y_te:  37×282       median pH, normalised   ← not used in this project
features:     1×11  feature names
location_ids: 37×1  station IDs
location_group: 1×3 three connected station groups (1, 8, 28 stations)
```

- **Days:** 705 contiguous days, 2016-01-28 to 2018-01-01.
- **Provided split:** 423 training days, 282 test days.
- **Unit of analysis:** station-day. `705 × 37 = 26,085`.
- **Values:** globally min–max normalised to approximately [0, 1]. There is no mapping back to physical units.

## Features

| Index | Short name | Family |
|---|---|---|
| 0 | `SC_max` | Specific conductance |
| 1 | `pH_max` | pH |
| 2 | `pH_min` | pH |
| 3 | `SC_min` | Specific conductance |
| 4 | `SC_mean` | Specific conductance |
| 5 | `DO_max` | Dissolved oxygen |
| 6 | `DO_mean` | Dissolved oxygen |
| 7 | `DO_min` | Dissolved oxygen |
| 8 | `T_mean` | Temperature |
| 9 | `T_min` | Temperature |
| 10 | `T_max` | Temperature |

## What not to do

- Do not use `Y_tr` / `Y_te`. They are the pH forecasting target and are outside the project scope.
- Do not apply drinking-water or regulatory thresholds. The values are normalised and the physical scale is unavailable.
- Do not claim that a flagged station-day is a pollution event or a confirmed sensor failure.
- Do not random-split the data. The series is temporally autocorrelated; use the provided boundary.

## Re-download

```bash
curl -L -o data/raw/water+quality+prediction-1.zip \
  "https://archive.ics.uci.edu/static/public/733/water+quality+prediction-1.zip"
unzip -o data/raw/water+quality+prediction-1.zip -d data/raw/
```

Verify the checksum in `raw/data_log.md` after downloading.
