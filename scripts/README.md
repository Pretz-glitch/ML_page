# `scripts/` — runnable experiments

Run in order, or use the one-command reproduction:

```bash
./scripts/run_all.sh
```

| Script | Phase | Status | Output |
|---|---|---|---|
| `01_build_dataset.py` | 1 | ✅ | `data/processed/station_day.csv`, `station_day_scaled.csv`, `scaler_params.csv` |
| `02_audit.py` | 1 | ✅ | audit tables, four figures, `results/phase1_summary.md` |
| `phase2_smoke.py` | 2 | ✅ | `results/tables/phase2_smoke.csv`, `results/phase2_smoke.md` |
| `03_run_detection.py` | 3 | ✅ | detection table, summary, paired comparisons, heatmap |
| `04_run_explanations.py` | 4 | ✅ | faithfulness table, attribution rows, paired, stability, heatmap |
| `05_real_record.py` | 5 | ✅ | real flags, flag/overlap/station/date summaries, case studies, figure |
| `06_release_check.py` | 6 | ✅ | `results/release_manifest.json` |
| `run_all.sh` | 6 | ✅ | full pipeline + release check + tests |

## Conventions

- Every modelling script accepts `--seed` and `--config` (or uses documented defaults).
- Every experiment script writes a JSON log with the git commit, package versions, seeds and checksums; the release check writes the manifest.
- Scripts never edit each other's outputs.
- Scripts do not contain analysis logic that belongs in `src/`.

## Individual commands

```bash
python scripts/01_build_dataset.py
python scripts/02_audit.py
python scripts/phase2_smoke.py
python scripts/03_run_detection.py
python scripts/04_run_explanations.py
python scripts/05_real_record.py
python scripts/06_release_check.py
pytest -q
```
