# `scripts/` — runnable experiments

Run in order. Each script writes its outputs to `results/` and prints a `Wrote ...` line.

| Script | Phase | Status | Output |
|---|---|---|---|
| `01_build_dataset.py` | 1 | ✅ | `data/processed/station_day.csv`, `station_day_scaled.csv`, `scaler_params.csv` |
| `02_audit.py` | 1 | ✅ | audit tables, four figures, `results/phase1_summary.md` |
| `03_run_detection.py` | 3 | ✅ | `results/tables/detection.csv`, `detection_summary.csv`, `detection_paired.csv`, heatmap |
| `04_run_explanations.py` | 4 | ⬜ | `results/tables/faithfulness.csv`, bar chart |
| `05_real_record.py` | 5 | ⬜ | `results/tables/real_flags.csv`, case figure |
| `06_make_figures.py` | 6 | ⬜ | final figures from saved CSVs |

## Phase 2 verification

Phase 2 has a dedicated integration script, `phase2_smoke.py`. It fits every detector on clean training data, injects each fault type into the test period, checks label/mask integrity, and writes `results/tables/phase2_smoke.csv` plus a JSON log. The curated report is [`results/phase2_smoke.md`](../results/phase2_smoke.md). Phase 3 adds the full `03_run_detection.py` benchmark.

## Conventions

- Every modelling script accepts `--seed` and `--config` (or uses documented defaults).
- Every script writes a JSON log with the git commit, package versions, seeds and checksums.
- Scripts never edit each other's outputs.
- Scripts do not contain analysis logic that belongs in `src/`.

## Current commands

```bash
python scripts/01_build_dataset.py
python scripts/02_audit.py
python scripts/phase2_smoke.py
python scripts/03_run_detection.py
pytest -q
```

## Later commands

```bash
python scripts/04_run_explanations.py --seeds 0 1 2 3 4
python scripts/05_real_record.py
python scripts/06_make_figures.py
```
