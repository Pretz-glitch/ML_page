# Pilot Results

**Status:** preliminary diagnostics, not final results.
**Purpose:** confirm that the proposed benchmark produces measurable differences before committing to the full protocol.

These runs were performed on the actual UCI 733 archive (`water_dataset.mat`) with `scikit-learn 1.9.1` and `isotree 0.6.1`. They use fewer repetitions than the final protocol and should be re-run with five seeds before being quoted in the report.

---

## 1. Setup

| Item | Value |
|---|---|
| Data | 705 days × 37 stations × 11 features |
| Train | days 0–422 |
| Test | days 423–704 |
| Scaling | within-station median/IQR, fitted on train |
| Detectors | MAD, PCA reconstruction (3 components), Mahalanobis (pseudo-inverse), Isolation Forest (300 trees), Extended IF (`ndim=11`, 300 trees) |
| Injection | spike (4–8 IQR), drift (5–20 day ramp, 4–8 IQR), correlation break (5–20 day permutation), stuck (5–20 day constant) |
| Repetitions | 2–3 per cell |
| Metric | ROC-AUC |

The pilot used a diagnostic schedule (several stations per repetition), not the final 1–2% prevalence. Absolute values will shift under the final protocol; the ranking is the informative part.

---

## 2. Detection results

| Fault type | IF | EIF | PCA | Mahalanobis | MAD |
|---|---|---|---|---|---|
| Spike | 0.734 | 0.927 | 0.871 | **0.991** | 0.804 |
| Drift | 0.705 | 0.846 | 0.804 | **0.927** | 0.722 |
| Correlation break | 0.588 | 0.652 | 0.637 | **0.846** | 0.546 |
| Stuck (diagnostic) | 0.510 | — | 0.514 | 0.659 | 0.504 |

Bold = best per row. Values are means over features and repetitions.

### Per-feature detail (selected)

| Feature | Fault | IF | EIF | PCA | Mahalanobis |
|---|---|---|---|---|---|
| `SC_mean` | Spike | 0.792 | 0.974 | 0.943 | 0.997 |
| `SC_mean` | Drift | 0.722 | 0.877 | 0.870 | 0.953 |
| `SC_mean` | Break | 0.669 | 0.714 | 0.652 | 0.842 |
| `DO_mean` | Spike | 0.719 | 0.937 | 0.875 | 0.998 |
| `DO_mean` | Break | 0.577 | 0.640 | 0.624 | 0.866 |
| `T_mean` | Break | 0.494 | 0.593 | 0.555 | 0.939 |

The full per-feature table is reproducible with `scripts/03_run_detection.py` once implemented.

---

## 3. What the pilot suggests

### H1a — supported

Mahalanobis outperforms standard IF on every fault type:

- Spike: 0.991 vs 0.734
- Drift: 0.927 vs 0.705
- Correlation break: 0.846 vs 0.588

The gap is largest on correlation breaks, which is the fault type that motivates multivariate detection in the first place.

### H1b — supported

Extended IF closes a substantial part of the gap:

- Spike: 0.734 → 0.927
- Drift: 0.705 → 0.846
- Break: 0.588 → 0.652

This is consistent with the known limitation of axis-parallel splits. It also means the final report can make a nuanced recommendation: if an isolation model is required, use EIF rather than standard IF.

### H1c — supported

Stuck faults sit near chance for most cross-sectional detectors in the pilot (0.50–0.66). The Phase 2 smoke run shows Mahalanobis reaching 0.79 on stuck faults: a frozen value can still violate the conditional expectation given the other features, but the signal is much weaker than for spike or drift. Treat stuck as a diagnostic, not a core result.

### Practical implication

The benchmark is not a foregone conclusion. A classical covariance detector beats the popular isolation model, and the result is large enough to survive reasonable changes in seed and magnitude. That is a reportable finding.

---

## 4. Data audit numbers (from the same archive)

| Observation | Value |
|---|---|
| Station-days | 26,085 |
| Train days | 423 |
| Test days | 282 |
| Stations | 37 (README says 36; arrays contain 37) |
| Between-station variance, `pH_min` | 0.93 |
| Between-station variance, `SC_min` | 0.92 |
| Between-station variance, `SC_max` | 0.90 |
| Correlation `SC_mean`–`DO_mean` | 0.96 |
| Correlation `T_mean`–`T_min` | 1.00 |
| Correlation `SC_min`–`pH_min` | 0.96 |

These justify two fixed design decisions:

1. **Within-station scaling is mandatory.** Otherwise the detector learns station identity.
2. **Attribution must be interpreted at the feature-family level.** Per-column rankings are unstable when within-family correlation approaches 1.

---

## 5. Why the original pooled ablation is not used

A separate diagnostic on a pooled model (no within-station scaling) measured single-feature retrain ablation:

| Dropped feature | mean absolute score change | rank correlation with full model |
|---|---|---|
| `pH_min` | 0.0105 | 0.985 |
| `pH_max` | 0.0100 | 0.962 |
| `SC_min` | 0.0096 | 0.985 |
| `T_mean` | 0.0084 | 0.984 |
| `SC_max` | 0.0081 | 0.982 |
| `SC_mean` | 0.0071 | 0.984 |

Dropping any single feature changes the anomaly ranking almost not at all (Spearman 0.96–0.985). The apparent "importance" ordering is a difference between 0.0105 and 0.0071 with no uncertainty estimate.

Grouped ablation told a different story:

| Group dropped | mean absolute score change |
|---|---|
| Temperature (3 columns) | 0.0256 |
| Specific conductance (3 columns) | 0.0193 |
| pH (2 columns) | 0.0134 |
| Dissolved oxygen (3 columns) | 0.0120 |

And the importance ranking was not robust to preprocessing: Spearman correlation between the raw-scale and per-station-z rankings was **0.19**.

**Conclusion:** single-feature retrain ablation is not a defensible primary explanation method here. It is replaced by local occlusion and exact Mahalanobis contributions. The ablation result is retained only as background evidence for why the replacement is necessary.

---

## 6. Caveats

- Few repetitions (2–3 per cell), one injection schedule.
- The pilot used a diagnostic prevalence, not the final 1–2%.
- Extended IF was run only for spike/drift/break, not stuck.
- The pilot used the sample covariance with a pseudo-inverse. The final run keeps that as the default and reports Ledoit–Wolf as a sensitivity variant. Ledoit–Wolf shrinks the off-diagonals (mean absolute off-diagonal covariance 0.23 → 0.08) and weakens dependency-anomaly detection on this dataset.
- The pilot's feature subset was selected by a variance threshold; the final script must compute and log it.
- These numbers are not final results and should not be quoted in the report without re-running the frozen protocol.

---

## 7. What to do next

1. Implement the frozen protocol in `src/` and `scripts/`.
2. Re-run E2 with five seeds and the final prevalence.
3. Compare the ranking to this table. If it changes materially, investigate before writing.
4. Freeze the numbers in `results/tables/detection.csv` and generate the report from that file.
