# Related Work and Positioning

This is a short, purpose-built review. It covers only the literature needed to justify the research questions and the design choices. Full-length background notes are kept outside the public repository.

---

## 1. Anomaly detection in water-quality data

Water-quality anomaly detection is an established application area, but most studies share three features: a single detector, an unlabelled record, and an evaluation based on plausibility rather than ground truth.

| Study | Method | Evaluation | Limitation for this project |
|---|---|---|---|
| Leigh et al. (2019), *STOTEN* | Feature-based framework for high-frequency sensor data | Expert inspection | Univariate features; no explanation faithfulness |
| Santos-Fernández et al. (2024), *Water Resources Research* | Bayesian/HMM spatio-temporal models | Simulation + case study | Focus on temporal/spatial autocorrelation, not attribution |
| El-Shafeiy et al. (2023), *Sensors* | MCN-LSTM | Detection metrics on labelled faults | Deep model; no explanation audit |
| Bhardwaj & Cenkeramaddi (2026), *IEEE Access* | Seasonal/trend residuals → Isolation Forest → K-Means | Expert-annotated events | IF assumed adequate; no detector comparison |
| Erb et al. (2026), *Water Research* | Isolation Forest + SHAP on flow-cytometry fingerprints | Indirect validation via Random Forest | SHAP used but not tested for faithfulness |
| Rigueira et al. (2026), *Water Resources Management* | Multitemporal decision trees + visual decision paths | F1 ≈ 99% on labelled windows | Supervised; different data modality |
| Fan & Fan (2026), *Advanced Electromagnetics* | Isolation Forest + Random Forest fusion | F1/AUC on threshold-derived labels | Label leakage risk; no attribution audit |

**Takeaway:** the field has plenty of "apply detector X to water data" papers. It has very few "does detector X actually beat a simple multivariate baseline, and does its explanation identify the cause?" papers.

---

## 2. Explainable anomaly detection

Anomaly detection has a mature explainability literature, but the water-quality literature has not adopted its evaluation standards.

| Work | Contribution | Relevance |
|---|---|---|
| Li, Zhu & van Leeuwen (2023), *ACM TKDD* | Survey and taxonomy of explainable anomaly detection | Defines the space; notes the lack of ground-truth evaluation |
| Carletti, Terzi & Susto (2023), *EAAI* | DIFFI: depth-based feature importance for Isolation Forest | The principled IF-specific attribution method |
| Arcudi et al. (2024), *EAAI* | ExIFFI + AUC-FS, an explanation-quality metric based on feature selection | Shows how to validate attributions indirectly |
| Antwarg et al. (2024), *AI* | Ground-truth framework for evaluating anomaly explanations | The methodological template for this project |
| Hooker, Mentch & Zhou (2021), *Statistics and Computing* | Permutation/ablation importance is unreliable with correlated features | Justifies replacing retrain ablation with local, conditional attribution |

**Takeaway:** explanation faithfulness is a solved *concept* but not a solved *practice* in water-quality applications. The tools exist; the audit does not.

---

## 3. Isolation Forest and its limits

| Work | Contribution |
|---|---|
| Liu, Ting & Zhou (2008, ICDM; 2012, *ACM TKDD*) | Original Isolation Forest |
| Hariri, Kind & Brunner (2019), *IEEE TKDE* | Extended Isolation Forest: hyperplane splits instead of axis-parallel splits |

The key limitation is structural. Standard IF partitions the feature space with cuts parallel to the axes. It is therefore good at finding points that are extreme in one or more coordinates, and weak at finding points that are ordinary in every coordinate but violate the dependence structure between coordinates. That second case is exactly the motivating example for multivariate water-quality anomaly detection: high temperature with high dissolved oxygen, or high conductance with low dissolved oxygen.

This is not a criticism of the original paper; it is a known property of the algorithm. It matters here because the water-quality literature often treats "multivariate input" as equivalent to "multivariate detection."

---

## 4. Data-quality faults vs environmental events

Sensor-data quality is a separate literature from water-quality event detection. The fault taxonomy used here is adapted from that work:

| Fault | Typical cause |
|---|---|
| Spike | Electrical transient, single bad reading |
| Drift | Calibration drift, biofouling, membrane ageing |
| Stuck | Firmware freeze, communication failure, flat-lined sensor |
| Correlation break | Channel swap, unit error, decoupled sensor |

The important distinction is that a data-quality fault is a property of the *record*, not of the *water*. A detector that cannot separate the two is not usable for either purpose.

Nahari et al. (2026), *IJACSA*, show a related failure mode: if anomalies are labelled by thresholding a variable that is also an input feature, the benchmark measures threshold recovery rather than anomaly detection. This project avoids that by generating labels through explicit perturbation of known rows and features, never through thresholds on input variables.

---

## 5. The gap

| What exists | What is missing |
|---|---|
| IF applied to water-quality data | A comparison against a strong multivariate baseline on the same faults |
| SHAP applied to IF flags | A test of whether SHAP (or any attribution) recovers the corrupted feature |
| Fault taxonomies in sensor-data quality | A fault taxonomy applied to multi-station water-quality records |
| Ground-truth explanation benchmarks in other domains | Application to water-quality anomaly detection |
| Detector comparisons on aggregate metrics | Per-fault-type comparisons that show *where* each detector fails |

This project fills the last four rows. It does not claim to fill the first two in general; it provides evidence on one dataset with a reproducible protocol.

---

## 6. Positioning statement

> Prior water-quality studies apply Isolation Forest and post-hoc attribution without checking whether the explanations identify the actual faulty measurement. This project builds a controlled fault-injection benchmark on 26,085 station-days and tests (i) whether standard IF is outperformed by a classical covariance detector, (ii) whether hyperplane isolation closes the gap, and (iii) whether post-hoc IF attributions recover the injected feature as reliably as an exact Mahalanobis decomposition.

The contribution is an **evaluation and a reusable benchmark**, not a new algorithm. That is appropriate for a single-person project with a one-week budget, and it is more defensible than claiming a novel detector that has not been compared to anything.

---

## 7. References

1. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM*. DOI: 10.1109/ICDM.2008.17
2. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2012). Isolation-based anomaly detection. *ACM TKDD*, 6(1). DOI: 10.1145/2133360.2133363
3. Hariri, S., Kind, M. C., & Brunner, R. J. (2019). Extended Isolation Forest. *IEEE TKDE*, 33(4). DOI: 10.1109/TKDE.2019.2947676
4. Carletti, M., Terzi, M., & Susto, G. A. (2023). Interpretable anomaly detection with DIFFI. *Engineering Applications of Artificial Intelligence*, 119. DOI: 10.1016/j.engappai.2022.105730
5. Arcudi, A., Frizzo, D., Masiero, C., & Susto, G. A. (2024). Enhancing interpretability and generalizability in extended isolation forests. *Engineering Applications of Artificial Intelligence*, 133. DOI: 10.1016/j.engappai.2024.109409
6. Li, Z., Zhu, Y., & van Leeuwen, M. (2023). A survey on explainable anomaly detection. *ACM TKDD*, 18(1). DOI: 10.1145/3609333
7. Leigh, C., et al. (2019). A framework for automated anomaly detection in high-frequency water-quality data from in situ sensors. *Science of the Total Environment*, 659. DOI: 10.1016/j.scitotenv.2019.02.085
8. Santos-Fernández, E., et al. (2024). Unsupervised anomaly detection in spatio-temporal stream network sensor data. *Water Resources Research*, 60. DOI: 10.1029/2023WR035707
9. Erb, I. K., et al. (2026). Beyond baselines for anomaly detection: Isolation Forest for baseline-free identification of anomalies in microbial water quality using explainable AI. *Water Research*. DOI: 10.1016/j.watres.2026.126825
10. Bhardwaj, J., & Cenkeramaddi, L. R. (2026). Unsupervised detection of multivariate water quality anomalies via residual feature embeddings. *IEEE Access*. DOI: 10.1109/ACCESS.2026.3704534
11. Rigueira, X., Olivieri, D. N., Araujo, M., & Pazo, M. (2026). An explainable AI model for anomaly detection in water quality data. *Water Resources Management*. DOI: 10.1007/s11269-026-04676-2
12. Fan, G. T., & Fan, D. K. (2026). Anomaly detection of water physicochemical indicators based on an Isolation Forest–Random Forest fusion algorithm. *Advanced Electromagnetics*, 15(3). DOI: 10.7716/aem.v15i3.4242
13. Antwarg, L., Galed, C., Rokach, L., & Shapira, B. (2024). Evaluating anomaly explanations using ground truth. *AI*, 5(4). DOI: 10.3390/ai5040117
14. Hooker, G., Mentch, L., & Zhou, S. (2021). Unrestricted permutation forces extrapolation: variable importance requires at least one more model. *Statistics and Computing*, 31. DOI: 10.1007/s11222-021-10057-z
15. Nahari, R. V., Prasetyo, T., & Alfita, R. (2026). Definitional label leakage in aquaculture water quality anomaly detection. *IJACSA*, 17(8). DOI: 10.14569/IJACSA.2026.0170873
16. Zhao, L., Gkountouna, O., & Pfoser, D. (2019). Spatial auto-regressive dependency interpretable learning based on spatial topological constraints. *ACM TSAS*, 5(3), Article 19. DOI: 10.1145/3339823

---

## 8. How to use this document

- In the report, use Section 5 (the gap) and Section 6 (positioning) almost verbatim.
- Do not overclaim novelty. "First" claims are risky; this project is likely the first *application* of a ground-truth explanation audit to this dataset, not the first anomaly detector for water quality.
- Cite the 2026 papers. They are the strongest evidence that the application niche is occupied and that the evaluation gap is real.
