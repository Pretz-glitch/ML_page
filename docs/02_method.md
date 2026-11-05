# Method

This document is the execution contract for the project. If an implementation choice is not specified here, it is a free choice; if it conflicts with this document, this document wins unless the decision log in `ROADMAP.md` is updated.

---

## 1. Overview

```
raw tensor (705, 37, 11)
        │
        ├── flatten ──► station-day table (26,085, 11)
        │
        ├── temporal split ──► train (423 days) / test (282 days)
        │
        ├── fit scaler on train ──► within-station median/IQR
        │
        ├── clean test ──────────────► detectors ──► scores
        │                                  │
        └── fault injection ───────────────┘
                    │
                    ▼
          detection metrics (AUC, AP, recall@k)
                    │
                    ▼
          local attribution ──► faithfulness metrics
```

---

## 2. Data construction

### 2.1 Load

`water_dataset.mat` contains:

| Variable | Shape | Meaning |
|---|---|---|
| `X_tr` | 1×423 cell | each cell is 37×11 (stations × features) for one day |
| `X_te` | 1×282 cell | each cell is 37×11 for one day |
| `Y_tr` | 37×423 | median pH (not used) |
| `Y_te` | 37×282 | median pH (not used) |
| `features` | 1×11 | feature names |
| `location_ids` | 37×1 | station IDs |
| `location_group` | 1×3 | three connected station groups |

Build a tensor:

```python
X = np.stack([X_tr[0, t] for t in range(423)] + [X_te[0, t] for t in range(282)], axis=0)
# shape (705, 37, 11)
```

### 2.2 Flatten

One row per station-day:

| Column | Type | Source |
|---|---|---|
| `day_index` | int | 0–704 |
| `date` | date | 2016-01-28 + `day_index` |
| `station_id` | int | `location_ids` |
| `station_group` | int | derived from `location_group` |
| `split` | str | `train` for days 0–422, `test` for 423–704 |
| `f1 … f11` | float | the 11 measurements |

Expected row count: `705 × 37 = 26,085`.

### 2.3 Feature names

Use short, stable names in code and figures:

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

Keep a `feature_family` mapping for grouped summaries and interpretation.

---

## 3. Preprocessing

### 3.1 Temporal split

Use the boundary provided by the dataset:

- **Train:** days 0–422 (2016-01-28 to 2017-03-25).
- **Test:** days 423–704 (2017-03-26 to 2018-01-01).

Never shuffle. Never compute any statistic on test data and apply it to train.

### 3.2 Within-station robust scaling (mandatory)

For each station `s` and feature `j`, using **training days only**:

```
median[s, j] = median(X_train[:, s, j])
iqr[s, j]    = percentile(X_train[:, s, j], 75) - percentile(..., 25)
scale[s, j]  = max(iqr[s, j], 1e-6)
z[t, s, j]   = (X[t, s, j] - median[s, j]) / scale[s, j]
```

Why: in the raw data, 90%+ of the variance in `pH_min`, `SC_min`, and `SC_max` is *between stations*. Without this step, a pooled detector mostly learns station identity.

If `iqr` is zero for a station-feature pair (possible for near-constant columns), fall back to the station's standard deviation; if that is also zero, set the scaled value to zero and exclude that feature from injection.

### 3.3 Optional seasonal residualisation (sensitivity only)

Fit, per station and feature, a linear model on training data:

```
x_t = a + b·t + c·sin(2πt/365) + d·cos(2πt/365) + ε_t
```

Use the residual `ε_t` as an alternative representation. Report this as a **sensitivity variant**, not the primary protocol. The primary protocol is 3.2.

Reason: the training period is 423 days, slightly more than one annual cycle, so the harmonic terms are estimable but not stable. Do not let a fragile seasonal model become the main result.

### 3.4 Missing and floor values

- Assert no `NaN` or `inf`.
- Report the fraction of values at the floor (`≤ 0.001`) per feature.
- Do **not** impute unless a feature is entirely missing for a station (it is not).
- If a station-feature pair has zero variance in training, exclude it from injection and note it.

---

## 4. Fault injection

Inject into the **test period only**, into the **raw** values, then apply the train-fitted scaler. This ensures the detector sees realistic scaled values and the scaler never sees corrupted data.

### 4.1 Eligible features

Only inject into features with meaningful within-station variance:

```
median_over_stations(iqr[s, j]) > 0.01
```

In the pilot, this selected eight features: `pH_max`, `SC_mean`, `DO_max`, `DO_mean`, `DO_min`, `T_mean`, `T_min`, `T_max`. Compute the list at runtime and record it; do not hard-code it without a check.

### 4.2 Fault types

#### Spike

- One station-day, one feature.
- `x' = clip(x ± k·iqr[s, j], 0, 1)`, with `k ∈ {4, 6, 8}` and random sign.
- Label the single row.

#### Drift

- One station, one feature, contiguous window of `w ∈ [5, 20]` days.
- `x'_t = clip(x_t + ramp_t, 0, 1)`, where `ramp` rises linearly from 0 to `±k·iqr[s, j]` over the window.
- Label all rows in the window.
- Detection delay is measured from the first affected day to the first flagged day.

#### Correlation break

- One station, one feature, contiguous window of `w ∈ [5, 20]` days.
- Replace the window's values with values of the same feature drawn from other station-days (without replacement), then clip to [0, 1].
- This preserves the marginal distribution of the feature but breaks its relationship with the others.
- Label all rows in the window.

#### Stuck (optional diagnostic)

- One station, one feature, contiguous window of `w ∈ [5, 20]` days.
- Hold the value constant at the value on the first day of the window.
- Label all rows.
- Expected result: cross-sectional detectors fail. This is a deliberate negative control, not a core result.

### 4.3 Prevalence and sampling

- Target prevalence: **1–2%** of test station-days (≈100–200 of 10,434).
- For spike faults, sample rows uniformly without replacement.
- For window faults, sample non-overlapping windows per station; do not let two injected faults overlap in the same station-feature pair.
- Use the same injected rows across all detectors for a given seed, so comparisons are paired.

### 4.4 Labels

Store a tidy label table:

| Column | Meaning |
|---|---|
| `row_id` | index into the flattened test table |
| `station_id`, `date` | location |
| `feature` | injected column index/name |
| `fault_type` | spike / drift / break / stuck |
| `window_start`, `window_end` | for temporal faults |
| `magnitude` | k or ramp size |
| `seed` | injection seed |

And a boolean vector aligned with the scored rows.

### 4.5 Sanity checks

- Clean rows must be byte-identical to the input.
- Exactly the intended rows are labelled.
- Injected values remain in [0, 1].
- Injection never touches training data.
- The prevalence matches the target within rounding.

---

## 5. Detectors

All detectors are fitted **once per seed on clean training data** and then used to score clean and corrupted test sets. Do not refit per injection.

### 5.1 MAD (univariate baseline)

On the within-station scaled data:

```
score(x) = max_j |z_j|
```

This is the multivariate extension of a per-feature robust z-score, but it is still a univariate decision: it flags a row if *any single* feature is extreme. It cannot detect a correlation break.

### 5.2 Mahalanobis distance (covariance baseline)

Fit robust location `μ` and covariance `Σ` on training rows:

```
d²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

Implementation notes:

- Use `sklearn.covariance.LedoitWolf` or `MinCovDet` to avoid a singular covariance.
- If using the raw sample covariance, use `np.linalg.pinv`.
- Score = `sqrt(d²)`.

This is the core multivariate baseline. It explicitly models the correlation structure that a correlation-break fault violates.

### 5.3 Isolation Forest (method under test)

```python
IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination="auto",
    random_state=seed,
    n_jobs=-1,
)
```

Score orientation: `sklearn` returns higher `score_samples` for more normal points, so use `score = -model.score_samples(X)`.

Do not tune hyperparameters on the test set. If tuning is needed, use a small grid on the training period with an unsupervised criterion (e.g., stability of the top-k set), and report it.

### 5.4 Extended Isolation Forest (optional, low cost)

```python
isotree.IsolationForest(
    ntrees=300,
    ndim=n_features,      # >1 enables hyperplane splits
    random_seed=seed,
    nthreads=-1,
)
```

Score with `model.predict(X)` (higher = more anomalous in `isotree`; verify orientation on a known outlier before trusting it).

Add this row if Day 3 finishes early. It is a strong result: it tests whether IF's weakness is the axis-parallel split or the isolation principle itself.

### 5.5 What not to include

- No deep autoencoders (data size and time budget do not justify them).
- No LOF/OCSVM in the core (three detectors are enough).
- No supervised classifier (there are no real labels).
- No tuning on injected test data.

---

## 6. Local attribution methods

The question is: *for a flagged row, which feature is responsible?*

### 6.1 Occlusion attribution (model-agnostic)

For a row `x` and feature `j`:

```
x_replaced = x with feature j set to the station's training median (0 in scaled space)
contribution_j = score(x) - score(x_replaced)
```

A large positive contribution means the feature's value is pushing the score up.

Notes:

- This is an **interventional** attribution: it asks what happens if the feature is reset, holding the others fixed. With correlated features, the intervention may create an implausible row. Report this limitation.
- Compute for all 11 features per explained row. Cost is trivial.
- Use the same method for IF and Mahalanobis, so the detector comparison is fair.

### 6.2 Exact Mahalanobis contribution

For a scaled row `z`:

```
c_j = z_j · (Σ⁻¹ z)_j
```

Then `Σ_j c_j = zᵀ Σ⁻¹ z = d²`. This is an exact additive decomposition of the squared distance into per-feature contributions.

- Use the same `Σ` as the detector.
- Report both signed contributions and absolute magnitudes.
- This is the reference for a "faithful" attribution: it is exact, not post-hoc.

### 6.3 Optional: DIFFI

Only if M4 is complete and there is time. DIFFI is a depth-based attribution for Isolation Forest (Carletti et al., 2023). An MIT-licensed reference implementation exists; if adapted, cite it and retain the licence notice.

Do not start DIFFI before the core faithfulness table exists.

### 6.4 Optional: grouped SHAP

Only if needed for the report. Use **grouped** SHAP over the four feature families, not per-column SHAP, because within-family correlations reach 0.95–1.00 and the independence assumption is badly violated.

---

## 7. Metrics

### 7.1 Detection

| Metric | Definition | Why |
|---|---|---|
| AUC-ROC | Ranking quality across all thresholds | Threshold-free comparison |
| AUC-PR (average precision) | Precision–recall trade-off at low prevalence | Appropriate for imbalanced faults |
| Recall@k | Fraction of injected rows in the top-k scores, `k =` number injected | Matches operational review capacity |
| Precision@k | Fraction of top-k rows that are injected | Cost of reviewing flags |
| FPR at threshold | Fraction of clean test rows above the 99th percentile of clean train scores | False-alarm rate at a fixed operating point |
| Detection delay | Days from first affected day to first flagged day | Only for drift/stuck |

Report per fault type and overall. Do not average across fault types without showing the per-type table.

### 7.2 Faithfulness

| Metric | Definition |
|---|---|
| Top-1 recovery | `argmax_j |contribution_j| == injected feature` |
| MRR | Mean of `1 / rank(injected feature)` |
| Mean rank | Average rank of the injected feature |
| Deletion drop | `score(x) - score(x with top-attributed feature reset)` |
| Random drop | Same, but reset a random feature (baseline) |
| Stability | Jaccard overlap of top-1 feature across seeds for the same injected row |

A faithful explainer should have high top-1 recovery, low mean rank, and a deletion drop substantially larger than the random drop.

### 7.3 Real-record analysis

No accuracy metrics are possible. Report:

- Number and percentage of flags per detector.
- Overlap between detectors (Jaccard).
- Flag concentration by station and by date.
- Three case studies with feature profiles against station reference statistics.

---

## 8. Experiment matrix

| ID | Experiment | Factors | Repeats | Output |
|---|---|---|---|---|
| E1 | Data audit | — | 1 | variance table, correlation figure, coverage figure |
| E2 | Detection | 3 detectors × 3 faults × 3 magnitudes | 5 seeds | `detection.csv`, heatmap |
| E3 | Faithfulness | 2 explainers × 3 faults | 5 seeds | `faithfulness.csv`, bar chart |
| E4 | Real record | 3 detectors × 1 period | 1 | `real_flags.csv`, case figure |

Optional rows: EIF in E2; Mahalanobis occlusion in E3; stuck diagnostic in E2.

---

## 9. Statistical protocol

- **Seeds:** 5 independent injection seeds (`0–4`). Detector random seeds are tied to the injection seed.
- **Paired design:** all detectors see the same injected rows within a seed.
- **Reporting:** mean ± SD across seeds.
- **Confidence intervals:** paired bootstrap over injected rows (1,000 resamples) for AUC/AP differences.
- **Tests:** Wilcoxon signed-rank for paired detector comparisons; report the effect size, not only the p-value.
- **No p-hacking:** the primary comparisons are fixed in advance (IF vs Mahalanobis; IF-occlusion vs Mahalanobis contribution). Everything else is exploratory and labelled as such.

---

## 10. Reproducibility

- Pin package versions in `requirements.txt`.
- Every script takes `--seed` and writes a JSON log with:
  - git commit hash
  - package versions
  - seed
  - input file checksums
  - output file checksums
- No manual edits to result tables. Figures are generated from saved CSVs.
- A fresh clone plus the raw data must reproduce the main tables in under 10 minutes.

---

## 11. Implementation pitfalls

1. **Do not fit the scaler on test data.** This is the most common leakage in this dataset.
2. **Do not use `Y`.** It is the forecasting target and is not a feature.
3. **Do not treat 705 as the sample size.** It is the number of days; the sample is 26,085 station-days.
4. **Do not use `score_samples` orientation blindly.** Verify that higher means more anomalous before computing AUC.
5. **Do not inject into near-constant features.** A stuck fault in `pH_min` is indistinguishable from normal behaviour.
6. **Do not clip after scaling.** Clip raw values to [0, 1], then scale.
7. **Do not compare detectors on different injected rows.** Use the same rows within a seed.
8. **Do not report a single seed.** IF is stochastic; five seeds is the minimum for a defensible claim.
9. **Do not apply TreeSHAP to Isolation Forest.** The score is an average path length, not an additive tree output; the assumptions do not hold.
10. **Do not describe real-record flags as pollution.** They are candidates for review.
