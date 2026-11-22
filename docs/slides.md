# Which Measurement Is Lying?

### Explainable Detection of Data-Quality Faults in Multi-Station Water-Quality Data

**Preetham** · November 2026 · v1.0

---

## 1. The problem

- Water-quality stations record **multivariate** daily observations.
- An unusual station-day can be:
  - a real **environmental event**, or
  - a **data-quality fault** in one measurement channel.
- Most papers apply one detector to an unlabelled record and report flags.
- Explanations are shown, but rarely **tested against a known cause**.

**Question:** when a detector flags a station-day, does its explanation identify the measurement that is actually wrong?

---

## 2. The dataset

UCI *Water Quality Prediction* (733), Georgia, USA.

| Property | Value |
|---|---|
| Period | 2016-01-28 → 2018-01-01 |
| Stations | 37 |
| Station-days | **26,085** |
| Features | 11 concurrent measurements |
| Split | 423 train / 282 test days |
| Missing | 0 |
| Values | globally normalised to ≈[0, 1] |

Features: specific conductance, pH, dissolved oxygen, temperature — each with min/max/mean variants.

**No ground-truth anomaly labels.**

---

## 3. Why station-aware scaling is mandatory

Share of variance that is **between stations**:

| Feature | Between-station share |
|---|---:|
| `pH_min` | **0.928** |
| `SC_min` | **0.920** |
| `SC_max` | **0.901** |
| `DO_max` | 0.443 |
| `T_mean` | 0.083 |

Three features are almost station fingerprints.

**Method:** within-station median/IQR scaling, fitted on training days only.

**Consequence:** an anomaly means “unusual for this station”, not “this station is different”.

---

## 4. Fault injection with known labels

Inject into the held-out test period, then scale.

| Fault | Definition |
|---|---|
| **Spike** | one station-day, one feature, ±4–8 IQR |
| **Drift** | 5–20 day ramp ending at ±4–8 IQR |
| **Correlation break** | 5–20 day window replaced by donors from other station-days |
| **Stuck** | 5–20 day constant (diagnostic) |

- 2% of test station-days.
- 8 eligible features; near-constant `SC_max`, `SC_min`, `pH_min` excluded.
- Labels: row, station-day, feature, fault type, window, seed.
- Same injected rows across detectors → paired comparisons.

---

## 5. Detectors and explanations

**Detectors**

| Family | Method |
|---|---|
| Univariate | MAD (max absolute z) |
| Covariance | Mahalanobis (sample covariance + pseudo-inverse) |
| Reconstruction | PCA (3 components) |
| Axis-parallel isolation | Isolation Forest (300 trees) |
| Hyperplane isolation | Extended IF (`isotree`) |

**Explanations**

- **Simple occlusion:** replace a feature with the station median.
- **Conditional occlusion:** replace it with `E[x_j | x_-j]` under a Gaussian reference.
- **Exact Mahalanobis:** `c_j = z_j (Σ⁻¹z)_j`, summing to `d²`.

---

## 6. RQ1 — Detection results

Mean ROC-AUC, 5 seeds:

| Detector | Spike | Drift | Break | Stuck |
|---|---:|---:|---:|---:|
| MAD | 0.813 | 0.714 | 0.571 | 0.533 |
| **Mahalanobis** | **0.990** | **0.957** | **0.859** | **0.745** |
| Isolation Forest | 0.759 | 0.700 | 0.599 | 0.550 |
| PCA | 0.884 | 0.823 | 0.654 | 0.546 |
| Extended IF | 0.934 | 0.864 | 0.665 | 0.565 |

**Mahalanobis wins every fault type.**

---

## 7. RQ1 — Three findings

1. **Standard IF is weaker than MAD on spikes** (0.759 vs 0.813), and only marginally above it on correlation breaks (0.599 vs 0.571).
2. **Extended IF closes much of the gap** (0.934 / 0.864 / 0.665). The problem is the axis-parallel split, not the isolation principle.
3. **AUC hides the review cost.** At 2% prevalence:
   - IF spike AUC 0.759 → recall@k **0.041**
   - Mahalanobis spike AUC 0.990 → recall@k **0.698**

---

## 8. RQ2 — Attribution faithfulness

Top-1 recovery of the injected feature:

| Method | Spike | Drift | Break |
|---|---:|---:|---:|
| IF + simple | 0.760 | 0.702 | 0.281 |
| IF + conditional | 0.558 | 0.579 | 0.292 |
| Mahalanobis + simple | 0.574 | 0.451 | 0.203 |
| **Mahalanobis + conditional** | **0.995** | **0.959** | **0.728** |
| Mahalanobis exact | 0.954 | 0.810 | 0.339 |
| EIF + simple | 0.911 | 0.830 | 0.364 |

Chance top-1 ≈ 0.09.

---

## 9. RQ2 — The baseline matters

- **Simple occlusion is an invalid intervention for Mahalanobis** under strong collinearity: it creates implausible rows.
- The same detector looks poor with simple occlusion (0.574 on spikes) and excellent with conditional occlusion (0.995).
- **IF prefers the simple baseline** (0.760 / 0.702 vs 0.558 / 0.579).
- **The exact decomposition is exact, but not always the most faithful** on breaks: 0.339 vs 0.728 for conditional occlusion.
- Deterministic methods are perfectly stable; IF majority agreement drops to 0.78 on breaks.

**Explanation quality is a property of the detector + intervention pair.**

---

## 10. Real-record analysis

10,434 test station-days, no labels.

| Detector | Flags | Rate |
|---|---:|---:|
| MAD | 214 | 2.05% |
| Mahalanobis | 294 | 2.82% |
| Isolation Forest | 182 | 1.74% |
| PCA | 318 | 3.05% |
| Extended IF | 264 | 2.53% |

- **559** flagged by any detector (5.36%).
- **39** flagged by all three primary detectors.
- Pairwise Jaccard **0.123–0.620**.
- Station 2203700: 99 flags, **0 from IF**, **0 consensus**.
- Top date 2018-01-01: 19 flags, **18 from IF** (boundary/regime caution).

---

## 11. Case studies

| Case | Date | Station | Flags | Top feature | Top \|z\| |
|---|---|---|---|---:|---:|
| 1 | 2017-06-15 | 2208493 | 5/5 | `pH_max` | 11.0 |
| 2 | 2017-09-12 | 2203950 | 5/5 | `DO_max` | 6.0 |
| 3 | 2017-12-09 | 2336526 | 5/5 | `DO_min` | 1.33 |

**Case 3 is the key example:** no eligible feature exceeds 1.33 IQR, yet all five detectors flag the combination. This is the multivariate pattern the project was built to examine.

**Caveat:** these are candidate data-quality faults, not confirmed pollution events.

---

## 12. Limitations and takeaway

**Limitations**

- Synthetic faults, not verified real anomalies.
- One magnitude (6) in the faithfulness study.
- Five seeds → coarse detection intervals.
- Gaussian conditional reference.
- Normalised values; no physical thresholds.
- One dataset and geography.
- Near-constant features remain in detector inputs.

**Takeaway**

- State the detector, threshold, occlusion baseline and cross-check.
- Report prevalence-aware metrics.
- If an isolation model is required, use Extended IF.
- Treat every flag as a candidate for review, not a diagnosis.

**Repository:** github.com/Pretz-glitch/ML_page
