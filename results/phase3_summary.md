# Phase 3 — Detection benchmark summary

**Status:** final Phase 3 results. Five seeds, three magnitudes for spike and drift, 2% prevalence.

## Setup

| Item | Value |
|---|---|
| Data | 26,085 station-days; train 0–422, test 423–704 |
| Detectors | MAD, Mahalanobis, Isolation Forest (primary); PCA reconstruction, Extended IF (secondary) |
| Faults | spike, drift, correlation break; stuck diagnostic |
| Magnitudes | 4, 6, 8 × training IQR (spike, drift); not applicable (break, stuck) |
| Prevalence | 2% of test station-days |
| Seeds | 0, 1, 2, 3, 4 |
| FPR threshold | 99th percentile of clean training scores per detector |
| CIs | Percentile bootstrap over the five seeds (1,000 resamples) |

## Mean ROC-AUC

| Detector | Spike | Drift | Correlation break | Stuck |
|---|---:|---:|---:|---:|
| MAD | 0.813 | 0.714 | 0.571 | 0.533 |
| Mahalanobis | 0.990 | 0.957 | 0.859 | 0.745 |
| Isolation Forest | 0.759 | 0.700 | 0.599 | 0.550 |
| PCA reconstruction | 0.884 | 0.823 | 0.654 | 0.546 |
| Extended IF | 0.934 | 0.864 | 0.665 | 0.565 |

## Mean AUC standard deviation across seeds

Full 95% confidence intervals are in `detection_summary.csv`.

| Detector | Spike | Drift | Correlation break | Stuck |
|---|---:|---:|---:|---:|
| MAD | 0.0092 | 0.0156 | 0.0561 | 0.0508 |
| Mahalanobis | 0.0035 | 0.0240 | 0.0176 | 0.0413 |
| Isolation Forest | 0.0085 | 0.0278 | 0.0483 | 0.0606 |
| PCA reconstruction | 0.0107 | 0.0258 | 0.0464 | 0.0422 |
| Extended IF | 0.0071 | 0.0221 | 0.0343 | 0.0433 |

## Mean recall@k (k = number injected)

| Detector | Spike | Drift | Correlation break | Stuck |
|---|---:|---:|---:|---:|
| MAD | 0.090 | 0.027 | 0.033 | 0.019 |
| Mahalanobis | 0.698 | 0.614 | 0.348 | 0.137 |
| Isolation Forest | 0.041 | 0.030 | 0.031 | 0.035 |
| PCA reconstruction | 0.159 | 0.065 | 0.036 | 0.014 |
| Extended IF | 0.211 | 0.078 | 0.046 | 0.018 |

## Best detector per fault type

- **Spike:** Mahalanobis — AUC 0.990
- **Drift:** Mahalanobis — AUC 0.957
- **Correlation break:** Mahalanobis — AUC 0.859
- **Stuck:** Mahalanobis — AUC 0.745

## FPR at the training 99th-percentile threshold

- MAD: 0.0205
- Mahalanobis: 0.0282
- Isolation Forest: 0.0171
- PCA reconstruction: 0.0305
- Extended IF: 0.0254

## Paired comparisons (AUC, primary detectors)

### Spike (magnitude 6)

| Comparison | Mean difference | 95% CI | p |
|---|---:|---|---:|
| iforest_vs_mad | -0.055 | [-0.061, -0.049] | 0.002 |
| mahalanobis_vs_iforest | +0.230 | [+0.221, +0.237] | 0.002 |
| mahalanobis_vs_mad | +0.176 | [+0.169, +0.183] | 0.002 |

### Drift (magnitude 6)

| Comparison | Mean difference | 95% CI | p |
|---|---:|---|---:|
| iforest_vs_mad | -0.016 | [-0.041, +0.009] | 0.208 |
| mahalanobis_vs_iforest | +0.257 | [+0.244, +0.269] | 0.002 |
| mahalanobis_vs_mad | +0.240 | [+0.216, +0.262] | 0.002 |

### Correlation break

| Comparison | Mean difference | 95% CI | p |
|---|---:|---|---:|
| iforest_vs_mad | +0.029 | [+0.018, +0.040] | 0.002 |
| mahalanobis_vs_iforest | +0.259 | [+0.223, +0.295] | 0.002 |
| mahalanobis_vs_mad | +0.288 | [+0.245, +0.331] | 0.002 |

### Stuck

| Comparison | Mean difference | 95% CI | p |
|---|---:|---|---:|
| iforest_vs_mad | +0.017 | [-0.004, +0.038] | 0.148 |
| mahalanobis_vs_iforest | +0.195 | [+0.167, +0.224] | 0.002 |
| mahalanobis_vs_mad | +0.212 | [+0.190, +0.235] | 0.002 |

## What the results show

1. **Mahalanobis leads on every fault type.** It reaches 0.990 on spikes, 0.957 on drift, 0.859 on correlation breaks and 0.745 on stuck faults.
2. **Standard Isolation Forest is weaker than MAD on spikes.** AUC 0.759 vs 0.813, and it is only marginally above MAD on correlation breaks (0.599 vs 0.571). The axis-parallel split is the likely reason.
3. **Extended IF recovers much of the gap.** AUC 0.934 on spikes vs 0.759 for standard IF, and 0.665 vs 0.599 on correlation breaks. The isolation principle is not the problem; the split geometry is.
4. **PCA reconstruction is a strong linear baseline.** It sits between MAD and Extended IF on spike and drift, and above standard IF on correlation breaks.
5. **AUC alone hides the low-prevalence problem.** Standard IF reaches AUC 0.759 on spikes but recall@k is only 0.041; Mahalanobis reaches 0.698. At 2% prevalence, a high AUC can still mean very few true faults in the review list.
6. **Stuck faults remain hard.** All detectors are between 0.533 and 0.745; even Mahalanobis is far below its spike/drift performance. This is a diagnostic, not a core result.
7. **The operating point drifts.** At the training 99th percentile, test FPR is 1.7–3.1% rather than 1%, indicating mild score drift between train and test.

## Caveats

- Five seeds give coarse confidence intervals. The paired bootstrap p-values are descriptive, not precise inferential tests.
- The benchmark uses injected faults, not verified real anomalies. Results describe detector behaviour under this fault model.
- Correlations among the 11 features remain high (up to 0.996), so per-column attribution in Phase 4 should be interpreted by family.
- The heatmap averages over the three magnitudes for spike and drift; the full per-magnitude table is `detection_summary.csv`.
