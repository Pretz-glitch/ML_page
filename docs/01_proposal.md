# Proposal

## Title

**Explainable Detection of Data-Quality Faults in Multi-Station Water-Quality Data: A Ground-Truth Benchmark**

Working name: *Anomaly Page*

---

## 1. Problem statement

Water-quality monitoring networks produce multivariate station-day records in which an unusual observation may reflect a real environmental event or a data-quality fault in a single measurement channel. Unsupervised detectors such as Isolation Forest are widely used to flag such observations, and post-hoc attribution methods are used to explain the flags, but it is unknown whether those explanations identify the measurement actually responsible.

This project addresses that gap by constructing a controlled fault-injection benchmark on 26,085 station-days from 37 monitoring stations in Georgia, USA. Spike, drift, and correlation-break faults are injected into held-out data with known labels, and three detectors — robust univariate deviation (MAD), Isolation Forest, and Mahalanobis distance — are compared on detection performance. Local attribution methods are then evaluated on whether they recover the injected measurement.

The project does **not** forecast pH, classify water as safe or unsafe, or claim that flagged observations are pollution events.

---

## 2. Research questions

### RQ1 — Detection

How accurately do MAD, Isolation Forest, and Mahalanobis distance detect injected spike, drift, and correlation-break faults in multivariate water-quality records?

### RQ2 — Explanation

When a detector flags an injected fault, do local attribution methods identify the measurement that was actually corrupted, and how does attribution faithfulness differ between Isolation Forest and Mahalanobis?

---

## 3. Hypotheses

These are stated before running the full experiment. The pilot in `05_pilot_results.md` gives preliminary support, but the final numbers come from the frozen protocol.

| ID | Hypothesis | Basis |
|---|---|---|
| H1a | Mahalanobis outperforms standard Isolation Forest on all three fault types, most strongly on correlation breaks. | IF uses axis-parallel splits; Mahalanobis models covariance directly. |
| H1b | Extended Isolation Forest closes part of the gap with Mahalanobis on correlation breaks. | Hyperplane splits can represent dependence. Tested only if the optional EIF row is run. |
| H1c | No cross-sectional detector reliably detects stuck-sensor faults. | A stuck value remains marginally plausible; detection requires temporal context. |
| H2a | IF occlusion attributions recover the injected feature less often than exact Mahalanobis contributions. | Post-hoc occlusion is affected by correlated substitutes; the exact decomposition is not. |
| H2b | Attribution faithfulness decreases as the injected feature's correlation with the remaining features increases. | Correlated features can absorb or redirect attribution. |

---

## 4. Contribution

### What is new

1. **A fault-injection benchmark for UCI 733.** A reusable generator for spike, drift, and correlation-break faults on multi-station water-quality records, with known feature and row labels.
2. **A detector comparison by fault type.** Evidence about which detector family is appropriate for which kind of data-quality fault, rather than a single aggregate score.
3. **An explanation-faithfulness audit.** A test of whether post-hoc anomaly explanations recover the injected feature, including a comparison against an exact Mahalanobis decomposition.
4. **A selection guide.** A short decision table for practitioners: detector, explainer, and the conditions under which each is appropriate.

### What is not claimed

- No new anomaly-detection algorithm.
- No claim that detected faults are pollution events.
- No claim that the benchmark covers all real sensor-fault modes.
- No causal inference.
- No forecasting improvement.

### Positioning

Water-quality anomaly-detection papers typically apply one detector to an unlabelled record and report flags or case studies. Some apply SHAP to explain the flags. Very few test the detector against a strong multivariate baseline, and none in this literature validate the explanation against a known corrupted feature. This project does both on the same dataset, with a protocol that can be reused.

---

## 5. Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset 733, *Water Quality Prediction* |
| Licence | CC BY 4.0 |
| Original paper | Zhao, Gkountouna & Pfoser (2019), ACM TSAS 5(3), Article 19 |
| Geography | 37 monitoring stations, Georgia, USA |
| Period | 2016-01-28 to 2018-01-01 (705 contiguous days) |
| Raw structure | `X_tr`: 1×423 cells, each 37×11; `X_te`: 1×282 cells, each 37×11 |
| Unit of analysis | station-day (26,085 rows) |
| Features | 11 concurrent measurements |
| Values | globally min–max normalised to approximately [0, 1] |
| Missing values | none reported; floor values near 0.001 require inspection |
| Provided split | 423 train days, 282 test days (temporal) |

### Feature families

The 11 columns are not independent. They form four interpretable families:

| Family | Columns |
|---|---|
| Specific conductance (SC) | `SC_max`, `SC_min`, `SC_mean` |
| pH | `pH_max`, `pH_min` |
| Dissolved oxygen (DO) | `DO_max`, `DO_mean`, `DO_min` |
| Temperature (T) | `T_mean`, `T_min`, `T_max` |

Within-family correlations reach 0.95–1.00. This matters for both detection and explanation: a model can substitute one column for another, so per-column attribution is inherently unstable.

### Variables deliberately excluded

- `Y_tr` / `Y_te` (median pH): this is the forecasting target of the original dataset. Using it would both leak information and pull the project back toward pH forecasting, which is outside scope.
- Station ID and date as model features: retained as metadata for grouping, plotting, and leakage checks only.

---

## 6. Scope

### In scope

- Unsupervised detection on concurrent multivariate measurements.
- Three fault types with known labels.
- Three detectors and two local attribution methods.
- Detection metrics and attribution-faithfulness metrics.
- A descriptive analysis of flags in the real unlabelled record.

### Out of scope

- Next-day pH forecasting.
- Normal / Watch / Elevated alert categories.
- Real-time or streaming detection.
- Dashboards, Power BI, or web applications.
- Deep learning architectures.
- Spatial or graph-based modelling.
- Causal inference.
- Water-safety, toxicity, or regulatory classification.
- Missing-data imputation as a research question.

---

## 7. Expected outputs

| Output | Form |
|---|---|
| Station-day dataset | `data/processed/station_day.csv` |
| Fault benchmark | `src/faults.py` + tests |
| Detection results | `results/tables/detection.csv`, heatmap |
| Faithfulness results | `results/tables/faithfulness.csv`, bar chart |
| Real-record analysis | flag table + 3 case studies |
| Report | 8–10 pages |
| Presentation | 10–12 slides |
| Reproducible repository | scripts, tests, README |

---

## 8. Ethical and interpretive position

- The dataset contains **normalised** measurements. No physical thresholds are applied, and no statement about drinking-water safety is made.
- A flagged station-day is a **candidate data-quality fault**, not a confirmed pollution event or sensor failure.
- Synthetic faults are a controlled approximation of real fault modes. Performance on them is evidence about detector behaviour, not a guarantee of field performance.
- The project reports negative results. If Isolation Forest underperforms, that is a finding, not a failure.
- The raw data are public and contain no personal information.

---

## 9. Timeline

See [ROADMAP.md](../ROADMAP.md). The core scope is sized for five working days, with a three-day fallback that still answers both research questions.
