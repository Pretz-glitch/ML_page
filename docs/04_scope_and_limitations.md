# Scope and Limitations

## 1. Scope

### In scope

| Area | Included |
|---|---|
| Data | UCI 733, 37 stations × 705 days, 11 concurrent measurements |
| Unit of analysis | Station-day (26,085 rows) |
| Split | Temporal: 423 train days, 282 test days |
| Preprocessing | Within-station robust scaling fitted on train only |
| Detection | MAD, Isolation Forest, Mahalanobis (optional Extended IF) |
| Faults | Spike, drift, correlation break (optional stuck diagnostic) |
| Explanations | Occlusion attribution, exact Mahalanobis contribution |
| Evaluation | AUC-ROC, AUC-PR, recall@k, FPR, top-1 recovery, MRR, deletion, stability |
| Real data | Descriptive flag analysis and three case studies |

### Out of scope

| Area | Excluded because |
|---|---|
| pH forecasting | Outside the project scope; a different task |
| Alert categories (Normal/Watch/Elevated) | Would duplicate the team project and requires unverifiable thresholds |
| Dashboard / Power BI | Not needed to answer either research question |
| Deep learning | Data size and time budget do not justify it |
| Spatial modelling | Would expand scope and overlap prior work |
| Causal inference | Observational, unlabelled data cannot support it |
| Water-safety classification | Values are normalised; no physical thresholds exist |
| Real-time / streaming | Offline benchmark only |
| Missing-data imputation | No missing values in the distributed dataset |
| Multi-fault simultaneous injection | Scope creep; single-fault benchmark first |

---

## 2. Limitations

### 2.1 No real anomaly labels

The dataset contains no verified anomaly annotations. This is the central limitation and the reason the project uses injected faults.

**Consequence:** all detection and faithfulness metrics are measured on synthetic faults. They describe detector behaviour under the benchmark's fault model, not field performance.

**Mitigation:** report the real-record analysis separately and label it descriptive. Do not compute accuracy, precision, or recall on real data.

### 2.2 Synthetic faults are an approximation

The three fault types are motivated by sensor-data-quality literature, but real faults are more varied: intermittent dropouts, unit conversions, biofouling, calibration steps, communication gaps, and multi-channel failures.

**Consequence:** a detector that performs well here may fail on a fault mode not represented in the benchmark.

**Mitigation:** state the taxonomy explicitly, keep the injector configurable, and describe the benchmark as a starting point rather than a complete fault model.

### 2.3 Normalised measurements

All 11 features are globally min–max normalised to approximately [0, 1]. There is no mapping back to physical units, and the pH values in particular are on an unknown scale.

**Consequence:** no drinking-water threshold, regulatory range, or physical plausibility check can be applied.

**Mitigation:** report everything in normalised units; never describe a flag as unsafe water.

### 2.4 Station confounding

In the raw data, 90%+ of the variance in `pH_min`, `SC_min`, and `SC_max` is between stations, not within them. A pooled detector can learn station identity instead of anomaly behaviour.

**Consequence:** without within-station scaling, the detector flags "which station", not "what happened".

**Mitigation:** within-station scaling is mandatory and fitted on training data only. The decision is recorded in the roadmap and cannot be silently changed.

### 2.5 Feature collinearity

Within-family correlations reach 0.95–1.00 (`T_mean`–`T_min` ≈ 1.00; `SC_mean`–`DO_mean` ≈ 0.96; `SC_min`–`pH_min` ≈ 0.96).

**Consequence:** per-column attribution is inherently unstable; a model can substitute one correlated column for another.

**Mitigation:** report attribution at the column level but interpret it with the family structure in mind. Include a grouped summary. Avoid claiming that a single column is uniquely responsible when its family partner carries the same information.

### 2.6 Single dataset, single geography

The data cover 37 stations in Georgia, USA, over 705 days. The station groups are unbalanced (1, 8, and 28 stations).

**Consequence:** conclusions may not transfer to other catchments, climates, or sensor networks.

**Mitigation:** describe the benchmark as a case study and release the injector so others can repeat it on other data.

### 2.7 Cross-sectional detectors cannot see stuck faults

A stuck value remains a plausible value at every time point. Detecting it requires temporal context (no change over time), which the core detectors do not model.

**Consequence:** if stuck faults are included, all core detectors will perform near chance.

**Mitigation:** treat stuck as an optional diagnostic that motivates future temporal work, not as a core result.

### 2.8 Occlusion attribution is interventional and can create implausible rows

Occlusion replaces one feature with the station median while holding the others fixed. With correlated features, the resulting row may be outside the normal data manifold.

**Consequence:** occlusion contributions are not a unique decomposition and can be order-dependent in principle.

**Mitigation:** use occlusion as a common, model-agnostic baseline across detectors, and compare it with the exact Mahalanobis decomposition. Report the limitation explicitly.

### 2.9 No hyperparameter tuning on real anomalies

There are no real labels, so detector hyperparameters cannot be tuned for real performance.

**Consequence:** the comparison is between default configurations, not between optimally tuned models.

**Mitigation:** use the same configuration across seeds and report it. Do not tune on injected test data. If any tuning is done, use an unsupervised stability criterion on training data and disclose it.

---

## 3. Claims matrix

| Claim | Supported? | How to phrase it |
|---|---|---|
| "Detector X detects fault type Y better than detector Z on this benchmark." | ✅ Yes | Report AUC/AP with CIs, per fault type |
| "Explainer X recovers the injected feature more often than explainer Y." | ✅ Yes | Report top-1 recovery and MRR with CIs |
| "Detector X is better in general." | ❌ No | Restrict to the benchmark and fault taxonomy |
| "These flags are pollution events." | ❌ No | "Candidate data-quality faults requiring review" |
| "This station has unsafe water." | ❌ No | Never claim this |
| "Feature X causes the anomaly." | ❌ No | "Feature X contributes to the model's score" |
| "The benchmark covers real sensor faults." | ❌ No | "A controlled approximation of three fault types" |
| "The method works on other water-quality networks." | ❌ No | "Evidence from one 37-station dataset" |

---

## 4. Threats to validity

| Threat | Type | Mitigation |
|---|---|---|
| Train/test leakage through scaling | Internal | Scaler fitted on train only; asserted in tests |
| Seasonality inflates false positives | Internal | Temporal split; within-station scaling; optional seasonal sensitivity |
| Station identity dominates | Internal | Within-station scaling; station-level diagnostics |
| Fault injection unrealistic | Construct | Literature-grounded taxonomy; configurable magnitudes; explicit limitation |
| Correlated features distort attribution | Construct | Family-level summary; exact Mahalanobis comparison |
| Single dataset | External | Case-study framing; released injector |
| Single seed | Statistical | Five seeds; mean ± SD; paired tests |
| Multiple comparisons | Statistical | Pre-specified primary comparisons; exploratory results labelled |
| Implementation bugs in injector | Internal | Unit tests on row indices, labels, and clean-row integrity |

---

## 5. What would change the conclusions

Be explicit about the conditions under which the main findings would not hold:

1. **A different fault model.** If faults are injected with much smaller magnitudes, all detectors may fall to chance and the ranking may change.
2. **A different preprocessing choice.** Global scaling instead of within-station scaling would make station identity dominant; results would not be comparable.
3. **A different dataset.** A single-site, high-frequency record would have different correlation structure and different fault detectability.
4. **A different detector configuration.** Heavily tuned IF could close part of the gap, though the axis-parallel limitation is structural.
5. **Real labels.** If verified fault annotations existed, the benchmark would not be necessary; the synthetic ranking would need external validation.

Report these as boundary conditions, not as excuses.
