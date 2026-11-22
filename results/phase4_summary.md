# Phase 4 — Attribution faithfulness summary

**Status:** final Phase 4 results. Five seeds, magnitude 6, 2% prevalence, three core fault types.

## Setup

| Item | Value |
|---|---|
| Data | 26,085 station-days; train 0–422, test 423–704 |
| Faults | spike, drift, correlation break |
| Magnitude | 6 × training IQR (spike, drift); not applicable (break) |
| Prevalence | 2% of test station-days |
| Seeds | 0, 1, 2, 3, 4 |
| Methods | 5 primary + 6 secondary |
| Chance top-1 | ≈ 0.09 (1 of 11 features) |
| CIs | Percentile bootstrap over the five seeds |

## Top-1 recovery (all injected rows)

| Method | Spike | Drift | Correlation break |
|---|---:|---:|---:|
| IF + simple | 0.760 | 0.702 | 0.281 |
| IF + conditional | 0.558 | 0.579 | 0.292 |
| Mahalanobis + simple | 0.574 | 0.451 | 0.203 |
| Mahalanobis + conditional | 0.995 | 0.959 | 0.728 |
| Mahalanobis exact | 0.954 | 0.810 | 0.339 |
| MAD + simple* | 0.754 | 0.672 | 0.225 |
| MAD + conditional* | 0.469 | 0.409 | 0.147 |
| PCA + simple* | 0.796 | 0.734 | 0.290 |
| PCA + conditional* | 0.587 | 0.597 | 0.304 |
| EIF + simple* | 0.911 | 0.830 | 0.364 |
| EIF + conditional* | 0.836 | 0.792 | 0.311 |

## Mean reciprocal rank

| Method | Spike | Drift | Correlation break |
|---|---:|---:|---:|
| IF + simple | 0.838 | 0.781 | 0.444 |
| IF + conditional | 0.736 | 0.743 | 0.509 |
| Mahalanobis + simple | 0.727 | 0.633 | 0.394 |
| Mahalanobis + conditional | 0.997 | 0.973 | 0.819 |
| Mahalanobis exact | 0.974 | 0.873 | 0.527 |
| MAD + simple* | 0.785 | 0.717 | 0.346 |
| MAD + conditional* | 0.607 | 0.557 | 0.298 |
| PCA + simple* | 0.848 | 0.797 | 0.444 |
| PCA + conditional* | 0.715 | 0.720 | 0.487 |
| EIF + simple* | 0.941 | 0.871 | 0.512 |
| EIF + conditional* | 0.902 | 0.880 | 0.531 |

## Detection rate and detected-only top-1

| Method | Spike detected | Spike top-1 | Drift detected | Drift top-1 | Break detected | Break top-1 |
|---|---:|---:|---:|---:|---:|---:|
| IF + simple | 3.5% | 0.198 | 2.6% | 0.100 | 2.9% | 0.087 |
| IF + conditional | 3.5% | 0.343 | 2.6% | 0.353 | 2.9% | 0.293 |
| Mahalanobis + simple | 94.1% | 0.605 | 80.6% | 0.540 | 45.2% | 0.369 |
| Mahalanobis + conditional | 94.1% | 0.996 | 80.6% | 0.998 | 45.2% | 0.916 |
| Mahalanobis exact | 94.1% | 0.970 | 80.6% | 0.913 | 45.2% | 0.538 |

Detected = score at or above the 99th percentile of clean training scores. IF detects only ~3% of injected rows at this threshold, so its detected-only numbers rest on a small, biased subset.

## Stability (majority top-1 agreement across five detector seeds)

| Method | Spike | Drift | Correlation break |
|---|---:|---:|---:|
| IF + simple | 0.916 | 0.886 | 0.780 |
| IF + conditional | 0.802 | 0.790 | 0.855 |
| Mahalanobis + simple | 1.000 | 1.000 | 1.000 |
| Mahalanobis + conditional | 1.000 | 1.000 | 1.000 |
| Mahalanobis exact | 1.000 | 1.000 | 1.000 |
| MAD + simple* | 1.000 | 1.000 | 1.000 |
| MAD + conditional* | 1.000 | 1.000 | 1.000 |
| PCA + simple* | 1.000 | 1.000 | 1.000 |
| PCA + conditional* | 1.000 | 1.000 | 1.000 |
| EIF + simple* | 1.000 | 0.997 | 0.979 |
| EIF + conditional* | 0.997 | 0.993 | 0.991 |

## Paired comparisons (top-1 recovery)

| Comparison | Fault | Mean difference | 95% CI |
|---|---|---:|---|
| mahalanobis_conditional_vs_mahalanobis_simple | Spike | +0.421 | [+0.391, +0.454] |
| mahalanobis_conditional_vs_mahalanobis_simple | Drift | +0.507 | [+0.476, +0.537] |
| mahalanobis_conditional_vs_mahalanobis_simple | Correlation break | +0.526 | [+0.491, +0.556] |
| mahalanobis_exact_vs_mahalanobis_conditional | Spike | -0.041 | [-0.054, -0.030] |
| mahalanobis_exact_vs_mahalanobis_conditional | Drift | -0.150 | [-0.170, -0.129] |
| mahalanobis_exact_vs_mahalanobis_conditional | Correlation break | -0.389 | [-0.418, -0.359] |
| if_conditional_vs_if_simple | Spike | -0.202 | [-0.234, -0.170] |
| if_conditional_vs_if_simple | Drift | -0.125 | [-0.157, -0.093] |
| if_conditional_vs_if_simple | Correlation break | +0.011 | [-0.019, +0.041] |
| mahalanobis_conditional_vs_if_conditional | Spike | +0.437 | [+0.409, +0.466] |
| mahalanobis_conditional_vs_if_conditional | Drift | +0.381 | [+0.352, +0.412] |
| mahalanobis_conditional_vs_if_conditional | Correlation break | +0.436 | [+0.404, +0.469] |
| eif_conditional_vs_if_conditional | Spike | +0.278 | [+0.246, +0.309] |
| eif_conditional_vs_if_conditional | Drift | +0.213 | [+0.185, +0.242] |
| eif_conditional_vs_if_conditional | Correlation break | +0.020 | [-0.003, +0.043] |

## What the results show

1. **Mahalanobis with conditional occlusion is the most faithful method overall.** Top-1 recovery is 0.995 on spikes, 0.959 on drift and 0.728 on correlation breaks; MRR is 0.997 / 0.973 / 0.819.
2. **The occlusion baseline changes the conclusion.** With simple median occlusion, Mahalanobis looks poor (0.574 / 0.451 / 0.203) because replacing a feature with the station median creates an implausible row under strong collinearity. This is an artefact of the intervention, not of the detector.
3. **Detectors prefer different baselines.** IF is more faithful with simple occlusion on spikes and drift (0.760 / 0.702 vs 0.558 / 0.579); on correlation breaks the two are similar (0.281 vs 0.292). EIF is strong with both (0.911 / 0.830 / 0.364 and 0.836 / 0.792 / 0.311).
4. **The exact Mahalanobis decomposition is strong but not the most faithful on breaks.** It reaches 0.954 / 0.810 / 0.339. Conditional occlusion beats it on every fault type, most strongly on correlation breaks (0.728 vs 0.339), because the additive decomposition spreads credit across correlated features.
5. **Correlation breaks remain the hardest case.** The best method reaches 0.728; the other methods range from 0.15 to 0.36. Dependency faults are genuinely harder to attribute than spikes or drift.
6. **Stability is high for deterministic methods and lower for IF.** Mahalanobis, PCA and MAD agree perfectly across detector seeds; IF simple agrees on 0.92 / 0.89 / 0.78 of rows, and IF conditional on 0.80 / 0.79 / 0.86. EIF stays at or above 0.979.
7. **The deletion test agrees with the ranking.** Relative deletion advantage is 0.78 / 0.72 / 0.49 for Mahalanobis conditional, 0.76 / 0.66 / 0.34 for exact, and 0.10 / 0.09 / 0.04 for IF simple. The conditional intervention was used for the deletion test for every method, so the comparison is not confounded by implausible rows.

## Caveats

- The benchmark uses injected faults, not verified real anomalies.
- Only magnitude 6 is used in Phase 4; Phase 3 showed that detection improves with magnitude, and attribution likely does too.
- The conditional reference is a single Gaussian fitted on training data. It is the natural intervention for Mahalanobis, but it is an external reference for IF, PCA and MAD.
- Five seeds give coarse confidence intervals; the aggregated tables include SD and 95% CIs.
- Detected-only numbers for IF are based on ~3% of injected rows and should not be compared directly with Mahalanobis.
- 12 station-days with more than one corrupted feature were excluded from attribution evaluation.
