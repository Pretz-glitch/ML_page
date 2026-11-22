# Anomaly Page

**Explainable detection of data-quality faults in multi-station water-quality data.**

This project asks a narrow question: when an unsupervised anomaly detector flags a station-day, can its explanation identify the measurement that is actually responsible?

It uses the UCI *Water Quality Prediction* dataset (37 monitoring stations in Georgia, USA; 705 days; 11 concurrent measurements), injects controlled faults with known labels, and compares detectors and local attribution methods on both detection accuracy and explanation faithfulness.

---

## Problem statement

Water-quality monitoring networks produce multivariate station-day records in which an unusual observation may reflect a real environmental event or a data-quality fault in a single measurement channel. Unsupervised detectors such as Isolation Forest are widely used to flag such observations, and post-hoc attribution methods are used to explain the flags, but it is unknown whether those explanations identify the measurement actually responsible. This project builds a controlled fault-injection benchmark on 26,085 station-days from 37 monitoring stations in Georgia, USA. Spike, drift, and correlation-break faults are injected into held-out data with known labels, and three detectors — robust univariate deviation (MAD), Isolation Forest, and Mahalanobis distance — are compared on detection performance. Local attribution methods are then evaluated on whether they recover the injected measurement.

The project does **not** forecast pH, classify water as safe or unsafe, or claim that flagged observations are pollution events.

---

## Research questions

**RQ1 — Detection.** How accurately do MAD, Isolation Forest, and Mahalanobis distance detect injected spike, drift, and correlation-break faults in multivariate water-quality records?

**RQ2 — Explanation.** When a detector flags an injected fault, do local attribution methods identify the measurement that was actually corrupted, and how does attribution faithfulness differ between Isolation Forest and Mahalanobis?

---

## Why this is different

Most water-quality anomaly-detection papers pick one detector, apply it to an unlabelled record, and report a plausible-looking set of flags. They rarely test whether the detector is better than a simple multivariate baseline, and they rarely test whether the explanation points at the right variable.

This project treats both as measurable questions:

| Common practice | This project |
|---|---|
| One detector, no comparison | Three detectors compared on the same injected faults |
| Unlabelled flags presented as findings | Ground-truth faults with known location and type |
| Explanations shown but not evaluated | Attribution recovery measured against the injected feature |
| Forecast/alerts as the end product | A detector/explainer selection guide |

The expected contribution is an **evaluation**, not a new algorithm: a reusable fault-injection benchmark and evidence about which detector/explainer pair to trust for which fault type.

---

## Planned pipeline

```
UCI 733 (.mat)
    │
    ▼
station-day table  (705 days × 37 stations × 11 features = 26,085 rows)
    │
    ▼
train / test split by date  (423 / 282 days)
    │
    ▼
within-station robust scaling  (fit on train only)
    │
    ├── clean test set ──────────────────────────┐
    │                                            │
    ▼                                            ▼
fault injection (spike, drift, correlation break)   detectors
    │                                            (MAD, IF, Mahalanobis)
    └──────────────► detection metrics ◄──────────┘
                            │
                            ▼
                 local attribution methods
              (occlusion, Mahalanobis contribution)
                            │
                            ▼
                 attribution faithfulness
```

---

## Repository layout

```
README.md                  this file
ROADMAP.md                 execution plan, milestones, cut list
requirements.txt           pinned dependencies
docs/
  01_proposal.md           problem, questions, hypotheses, contribution
  02_method.md             data protocol, faults, detectors, metrics
  03_related_work.md       literature and positioning
  04_scope_and_limitations.md
  05_pilot_results.md      preliminary evidence from the real dataset
data/
  README.md                source, licence, structure
  raw/                     UCI archive and source log
src/                       pipeline modules (Phases 1–2 implemented)
scripts/                   runnable experiment scripts (Phases 1–2 implemented)
tests/                     unit tests (Phases 1–2 implemented)
results/                   generated tables and figures (logs are gitignored)
```

---

## Status

**Phase 5 complete.** Real-record analysis is implemented and verified. Phase 6 (report and release) is next.

| Phase | Status | Evidence |
|---|---|---|
| 1 — Data pipeline and audit | ✅ Complete | [results/phase1_summary.md](results/phase1_summary.md) |
| 2 — Detectors and fault injection | ✅ Complete | [results/phase2_smoke.md](results/phase2_smoke.md) |
| 3 — Detection benchmark | ✅ Complete | [results/phase3_summary.md](results/phase3_summary.md) |
| 4 — Explanation faithfulness | ✅ Complete | [results/phase4_summary.md](results/phase4_summary.md) |
| 5 — Real-record analysis | ✅ Complete | [results/phase5_summary.md](results/phase5_summary.md) |
| 6 — Report and release | ⬜ Next | — |

### Phase 1 at a glance

| Check | Result |
|---|---|
| Station-days | 26,085 (705 days × 37 stations) |
| Features | 11 |
| Split | 423 train / 282 test days |
| Missing values | 0 |
| Between-station variance ≥ 50% | `pH_min` (0.93), `SC_min` (0.92), `SC_max` (0.90) |
| Strongest correlation | `T_mean`–`T_min`, r = 0.996 |

### Phase 2 at a glance

| Check | Result |
|---|---|
| Detectors | MAD, Mahalanobis, PCA reconstruction, Isolation Forest, Extended IF |
| Fault types | spike, drift, correlation break, stuck |
| Injection | 2% prevalence; 209–225 labelled station-days per fault type |
| Outputs | corrupted tensor + tidy label table + cell and station-day masks |
| Tests | 56 passing |
| Smoke run | Mahalanobis 0.994 / 0.979 / 0.863 on spike / drift / correlation break |

### Phase 3 at a glance

| Check | Result |
|---|---|
| Runs | 5 seeds × 8 conditions × 5 detectors = **200 result rows** |
| Primary detectors | MAD, Mahalanobis, Isolation Forest |
| Secondary detectors | PCA reconstruction, Extended IF |
| Best detector | **Mahalanobis on every fault type** (AUC 0.990 / 0.957 / 0.859 / 0.745) |
| Standard IF | 0.759 / 0.700 / 0.599 / 0.550 — weaker than MAD on spikes |
| Extended IF | 0.934 / 0.864 / 0.665 / 0.565 — closes much of the isolation gap |
| Tests | 67 passing (11 new) |
| Outputs | `detection.csv`, `detection_summary.csv`, `detection_paired.csv`, heatmap |

### Phase 4 at a glance

| Check | Result |
|---|---|
| Attribution methods | 5 primary (IF simple/conditional, Mahalanobis simple/conditional/exact) + 6 secondary |
| Faults | spike, drift, correlation break |
| Seeds | 5 |
| Best top-1 | **Mahalanobis + conditional occlusion: 0.995 / 0.959 / 0.728** |
| Key contrast | Simple occlusion makes Mahalanobis look poor (0.574 / 0.451 / 0.203) because it creates implausible rows; the baseline choice changes the conclusion |
| Stability | IF simple 0.92 / 0.89 / 0.78; EIF ≥ 0.979; deterministic methods 1.0 |
| Tests | 79 passing (12 new) |
| Outputs | `faithfulness.csv`, `attribution_rows.csv`, `faithfulness_paired.csv`, `faithfulness_stability.csv`, heatmap |

### Phase 5 at a glance

| Check | Result |
|---|---|
| Test station-days | 10,434 |
| Flag rates | 1.74%–3.05% across detectors |
| Flagged by any detector | 559 (5.36%) |
| Flagged by all three primary | 39 |
| Flagged by at least two | 311 |
| Pairwise Jaccard | 0.123–0.620 |
| Top station | 2203700: 99 flags, but 0 from IF and 0 consensus |
| Cases | pH_max shock; DO_max shock; a multivariate DO/T case with no feature above 1.33 IQR |
| Tests | 88 passing (9 new) |
| Outputs | `real_flags.csv`, flag/overlap/station/date summaries, `case_studies.png` |

Reproduce with:

```bash
python scripts/01_build_dataset.py
python scripts/02_audit.py
python scripts/phase2_smoke.py
python scripts/03_run_detection.py
python scripts/04_run_explanations.py
python scripts/05_real_record.py
pytest -q
```


---

## Data

- **Source:** UCI Machine Learning Repository, *Water Quality Prediction* (dataset 733), CC BY 4.0.
- **Original paper:** Zhao, Gkountouna & Pfoser (2019), *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints*, ACM TSAS 5(3), Article 19.
- **Shape used here:** 705 days × 37 stations × 11 features = 26,085 station-days.
- **Values:** globally min–max normalised to approximately [0, 1]. There is no mapping back to physical pH, so no drinking-water thresholds are applied.

See [data/README.md](data/README.md) for the file inventory and checksum.

---

## Non-goals

- Next-day pH forecasting
- Normal / Watch / Elevated alert categories
- Real-time monitoring or dashboards
- Deep learning
- Causal inference
- Water-safety or toxicity classification

---

## Reproduction

Phase 1:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_build_dataset.py
python scripts/02_audit.py
pytest -q
```

Later phases add:

```bash
python scripts/03_run_detection.py
python scripts/04_run_explanations.py
python scripts/05_real_record.py
```

Every script writes its outputs to `results/` and ends with a `Wrote ...` line.
