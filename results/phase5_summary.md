# Phase 5 — Real-record flag analysis

**Status:** descriptive analysis of the unlabelled test period. No ground-truth labels.

## Setup

| Item | Value |
|---|---|
| Test period | 2017-03-26 to 2018-01-01 |
| Station-days | 10,434 |
| Detectors | MAD, Mahalanobis, Isolation Forest (primary); PCA, Extended IF (secondary) |
| Threshold | 99% of clean training scores, averaged over 5 seeds |
| Flag | score at or above that threshold |
| Case features | 8 eligible features; near-constant SC_max, SC_min and pH_min are excluded |
| Case studies | 3 |

## Flag counts

| Detector | Role | Flags | Rate | Stations | Dates | Median flagged score |
|---|---|---|---|---|---|---|
| MAD | primary | 214 | 2.05% | 22 | 125 | 8.4956 |
| Mahalanobis | primary | 294 | 2.82% | 26 | 176 | 10.4433 |
| Isolation Forest | primary | 182 | 1.74% | 25 | 78 | 0.6155 |
| PCA reconstruction | secondary | 318 | 3.05% | 22 | 165 | 19.5525 |
| Extended IF | secondary | 264 | 2.53% | 27 | 151 | 0.5762 |

Across detectors, flag rates range from **1.74%** to **3.05%**. A total of **559** station-days (5.36%) are flagged by at least one detector.

## Pairwise overlap

| Detector A | Detector B | Type | Intersection | Union | Jaccard | Agreement |
|---|---|---|---|---|---|---|
| MAD | Extended IF | primary-secondary | 183 | 295 | 0.620 | 0.989 |
| Mahalanobis | Extended IF | primary-secondary | 197 | 361 | 0.546 | 0.984 |
| Mahalanobis | PCA reconstruction | primary-secondary | 208 | 404 | 0.515 | 0.981 |
| PCA reconstruction | Extended IF | secondary-secondary | 192 | 390 | 0.492 | 0.981 |
| MAD | Mahalanobis | primary-primary | 151 | 357 | 0.423 | 0.980 |
| MAD | PCA reconstruction | primary-secondary | 151 | 381 | 0.396 | 0.978 |
| Isolation Forest | Extended IF | primary-secondary | 83 | 363 | 0.229 | 0.973 |
| MAD | Isolation Forest | primary-primary | 61 | 335 | 0.182 | 0.974 |
| Isolation Forest | PCA reconstruction | primary-secondary | 56 | 444 | 0.126 | 0.963 |
| Mahalanobis | Isolation Forest | primary-primary | 52 | 424 | 0.123 | 0.964 |

Pairwise Jaccard ranges from **0.123** to **0.620**. Only **39** station-days are flagged by all three primary detectors; **311** are flagged by at least two detectors.

## Concentration by station

| Station | Group | Rows | Any flag | Rate | MAD | Mahalanobis | IF | PCA | EIF |
|---|---|---|---|---|---|---|---|---|---|
| 2203700 | 2 | 282 | 99 | 35.11% | 66 | 96 | 0 | 99 | 84 |
| 21989793 | 1 | 282 | 66 | 23.40% | 18 | 29 | 27 | 56 | 25 |
| 2208493 | 2 | 282 | 44 | 15.60% | 30 | 17 | 28 | 19 | 24 |
| 2336240 | 2 | 282 | 41 | 14.54% | 15 | 16 | 0 | 37 | 22 |
| 2198840 | 1 | 282 | 30 | 10.64% | 25 | 7 | 12 | 11 | 19 |

The most-flagged station (2203700) accounts for **99** flagged station-days, **17.7%** of all flagged rows. It is flagged 66 times by MAD, 96 by Mahalanobis, 0 by Isolation Forest, 99 by PCA and 84 by Extended IF. A high flag count without cross-detector consensus is a station-level disagreement, not a confirmed fault.

## Concentration by date

| Date | Rows | Any flag | Rate | MAD | Mahalanobis | IF | PCA | EIF |
|---|---|---|---|---|---|---|---|---|
| 2018-01-01 | 37 | 19 | 51.35% | 2 | 1 | 18 | 2 | 2 |
| 2017-12-10 | 37 | 13 | 35.14% | 3 | 2 | 11 | 4 | 4 |
| 2017-09-12 | 37 | 13 | 35.14% | 7 | 7 | 9 | 7 | 10 |
| 2017-08-09 | 37 | 11 | 29.73% | 3 | 5 | 8 | 3 | 5 |
| 2017-12-09 | 37 | 10 | 27.03% | 7 | 3 | 8 | 5 | 6 |

The most-flagged date (2018-01-01) has **19** flagged station-days. Isolation Forest contributes **18** of them. A single date with many flags can reflect a shared regime shift, a boundary effect, or a system-wide event rather than independent faults.

## Case studies

| Case | Date | Station | Flags | Primary | Top feature | Top |z| |
|---|---|---|---|---|---|---|
| 1 | 2017-06-15 | 2208493 | 5/5 | 3/3 | pH_max | 11.00 |
| 2 | 2017-09-12 | 2203950 | 5/5 | 3/3 | DO_max | 6.00 |
| 3 | 2017-12-09 | 2336526 | 5/5 | 3/3 | DO_min | 1.33 |

The figure shows, for each case, the station's history of the most extreme feature and the full within-station z-profile at the flagged station-day.

Case 3 is a **multivariate flag**: its largest eligible-feature deviation is only 1.33 IQR, yet all five detectors flag the row. This is the combination-driven pattern the project was designed to examine.

## What this analysis can and cannot say

- These are **candidate data-quality faults**. There is no ground truth, so a flag is not a confirmed sensor failure, pollution event, or unsafe-water observation.
- A flag can reflect a real environmental event, a sensor fault, a seasonal shift, or a model artefact. This analysis cannot distinguish them.
- The values are normalised, so no physical threshold or regulatory classification is applied.
- The threshold is a design choice. The 99th percentile of training scores gives flag rates above 1% on test because the test period drifts slightly from training.
- Cross-detector agreement is modest: Jaccard ranges from 0.123 to 0.620. Consensus flags are a small, conservative subset.

## Caveats

- Unlabelled data: no precision, recall or F1 can be computed.
- The detectors are averaged over five seeds for stochastic models; deterministic detectors are repeated identically.
- Station confounding is reduced by within-station scaling but not eliminated.
- Near-constant features (`SC_max`, `SC_min`, `pH_min`) can still influence detector scores. Case studies and profile plots are restricted to the 8 features with meaningful within-station variation.
- The case studies are selected for consensus and eligible-feature severity, not because they are known faults.
