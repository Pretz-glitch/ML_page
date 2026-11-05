# `scripts/` — runnable experiments

Run in order. Each script writes its outputs to `results/` and prints a `Wrote ...` line.

| Script | Milestone | Output |
|---|---|---|
| `01_build_dataset.py` | M1 | `data/processed/station_day.csv` |
| `02_audit.py` | M1 | audit tables and figures |
| `03_run_detection.py` | M3 | `results/tables/detection.csv`, heatmap |
| `04_run_explanations.py` | M4 | `results/tables/faithfulness.csv`, bar chart |
| `05_real_record.py` | M5 | `results/tables/real_flags.csv`, case figure |
| `06_make_figures.py` | M6 | final figures from saved CSVs |

## Conventions

- Every script accepts `--seed` and `--config` (or uses documented defaults).
- Every script writes a JSON log with the git commit, package versions, seeds and checksums.
- Scripts never edit each other's outputs.
- Scripts do not contain analysis logic that belongs in `src/`.

## Planned commands

```bash
python scripts/01_build_dataset.py
python scripts/02_audit.py
python scripts/03_run_detection.py --seeds 0 1 2 3 4
python scripts/04_run_explanations.py --seeds 0 1 2 3 4
python scripts/05_real_record.py
python scripts/06_make_figures.py
```
