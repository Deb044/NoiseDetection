# Final Report: Tabular Noise Detection

## 1. Objective
The goal of this module is to identify and analyze noisy or anomalous entries in tabular datasets using both statistical and model-based techniques.

## 2. Folder Structure
The Tabular module follows the same high-level structure as the TimeSeries module:

- notebooks: step-wise experiments and demonstrations
- src: reusable implementation code
- report: project documentation and final summary
- requirements.txt: dependency list

## 3. Implemented Methods

### 3.1 IQR-Based Detection
- Notebook: notebooks/01_iqr.ipynb
- Source function: src/detectors.py -> detect_outliers_iqr
- Purpose: detect univariate outliers using quartile bounds

### 3.2 Z-Score Detection
- Notebook: notebooks/02_zscore.ipynb
- Source function: src/detectors.py -> detect_outliers_zscore
- Purpose: detect extreme values using standard deviation distance from mean

### 3.3 DBSCAN-Based Noise Detection
- Notebook: notebooks/03_dbscan.ipynb
- Source function: src/detectors.py -> detect_noise_dbscan
- Purpose: detect points not belonging to any dense cluster (label -1)

### 3.4 Isolation Forest (Model-Based)
- Notebook: notebooks/04_model_based.ipynb
- Source function: src/detectors.py -> detect_noise_isolation_forest
- Purpose: isolate rare observations using random partitioning

### 3.5 Visualization-Based Inspection
- Notebook: notebooks/05_visualisation.ipynb
- Source functions: src/visualization.py
- Plots included:
	- Boxplot for univariate spread and outliers
	- Histogram with Z-score boundaries
	- Pairplot for multivariate anomaly patterns

## 4. Evaluation Utilities
The src/evaluation.py module provides:

- summarize_labels: counts and percentage of detected noise labels
- compare_detection_methods: side-by-side comparison of methods by detection rate

## 5. Recommended Workflow
1. Start with visual inspection (notebooks/05_visualisation.ipynb).
2. Apply IQR and Z-score for quick univariate screening.
3. Apply DBSCAN for density-based anomalies.
4. Apply Isolation Forest for model-based anomaly detection.
5. Compare outputs using src/evaluation.py utilities and domain knowledge.

## 6. Key Observations
- Statistical methods are simple and interpretable but may miss contextual anomalies.
- DBSCAN is effective for non-linear cluster shapes but sensitive to eps and min_samples.
- Isolation Forest handles multivariate behavior well but depends on contamination setting.
- Visual diagnostics are useful for validating model findings.

## 7. Limitations
- No single method is universally best across all tabular distributions.
- Threshold and hyperparameter selection requires domain and data understanding.
- Ground-truth anomaly labels are often unavailable, so evaluation is typically indirect.

## 8. Multi-Dataset Benchmark Results (Notebook 06)

The benchmark results below were added from `Tabular/notebooks/06_dataset_benchmarking.ipynb` using the exported file `Tabular/report/benchmark_results.csv`.

### 8.1 Method-Wise Average Performance (Across 5 Datasets)

| Method | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.6227 | 0.7992 | 0.6345 | 0.6467 |
| Z-Score | 0.3206 | 0.8580 | 0.3909 | 0.3066 |
| IQR | 0.1074 | 0.9039 | 0.1811 | 0.1014 |
| DBSCAN | 0.0857 | 0.9353 | 0.1483 | 0.0836 |

### 8.2 Best Method Per Dataset (By F1)

| Dataset | Evaluation Mode | Best Method | F1 |
|---|---|---|---:|
| Pima Indians Diabetes | synthetic_injection | Isolation Forest | 0.9870 |
| Chronic Kidney Disease | synthetic_injection | Isolation Forest | 1.0000 |
| Wisconsin Breast Cancer | synthetic_injection | Isolation Forest | 0.9825 |
| Stroke Prediction | labeled | Z-Score | 0.1994 |
| Credit Card Fraud | labeled | Isolation Forest | 0.0566 |

### 8.3 Detailed Results

| Dataset | Method | Mode | Samples | Detected | Precision | Recall | F1 | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Pima Indians Diabetes | IQR | synthetic_injection | 768 | 154 | 0.2468 | 1.0000 | 0.3958 | 0.2468 |
| Pima Indians Diabetes | Z-Score | synthetic_injection | 768 | 44 | 0.8409 | 0.9737 | 0.9024 | 0.8201 |
| Pima Indians Diabetes | DBSCAN | synthetic_injection | 768 | 165 | 0.2303 | 1.0000 | 0.3744 | 0.2303 |
| Pima Indians Diabetes | Isolation Forest | synthetic_injection | 768 | 39 | 0.9744 | 1.0000 | 0.9870 | 1.0000 |
| Chronic Kidney Disease | IQR | synthetic_injection | 400 | 400 | 0.0500 | 1.0000 | 0.0952 | 0.0500 |
| Chronic Kidney Disease | Z-Score | synthetic_injection | 400 | 338 | 0.0592 | 1.0000 | 0.1117 | 0.0592 |
| Chronic Kidney Disease | DBSCAN | synthetic_injection | 400 | 400 | 0.0500 | 1.0000 | 0.0952 | 0.0500 |
| Chronic Kidney Disease | Isolation Forest | synthetic_injection | 400 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Wisconsin Breast Cancer | IQR | synthetic_injection | 569 | 231 | 0.1212 | 1.0000 | 0.2162 | 0.1212 |
| Wisconsin Breast Cancer | Z-Score | synthetic_injection | 569 | 50 | 0.5600 | 1.0000 | 0.7179 | 0.5600 |
| Wisconsin Breast Cancer | DBSCAN | synthetic_injection | 569 | 449 | 0.0624 | 1.0000 | 0.1174 | 0.0624 |
| Wisconsin Breast Cancer | Isolation Forest | synthetic_injection | 569 | 29 | 0.9655 | 1.0000 | 0.9825 | 0.9988 |
| Stroke Prediction | IQR | labeled | 5110 | 1184 | 0.1157 | 0.5502 | 0.1912 | 0.0856 |
| Stroke Prediction | Z-Score | labeled | 5110 | 784 | 0.1314 | 0.4137 | 0.1994 | 0.0829 |
| Stroke Prediction | DBSCAN | labeled | 5110 | 2076 | 0.0829 | 0.6908 | 0.1480 | 0.0723 |
| Stroke Prediction | Isolation Forest | labeled | 5110 | 256 | 0.1445 | 0.1486 | 0.1465 | 0.0965 |
| Credit Card Fraud | IQR | labeled | 284807 | 138473 | 0.0034 | 0.9695 | 0.0069 | 0.0034 |
| Credit Card Fraud | Z-Score | labeled | 284807 | 37816 | 0.0117 | 0.9024 | 0.0232 | 0.0108 |
| Credit Card Fraud | DBSCAN | labeled | 284807 | 150365 | 0.0032 | 0.9858 | 0.0064 | 0.0032 |
| Credit Card Fraud | Isolation Forest | labeled | 284807 | 14241 | 0.0293 | 0.8476 | 0.0566 | 0.1384 |

### 8.4 Interpretation of Results

1. Isolation Forest is the most reliable overall method in this benchmark.
It has the highest average F1 (0.6345) and PR-AUC (0.6467), and it is the best method on 4 out of 5 datasets.

2. Z-Score is the strongest statistical baseline.
It performs well on synthetic-injection datasets and is the best method for Stroke Prediction in labeled evaluation, but it remains below Isolation Forest in overall consistency.

3. IQR and DBSCAN show very high recall but low precision.
Both methods mark a large number of samples as anomalies in several datasets, which increases false positives and lowers F1.

4. Performance depends strongly on dataset properties and labeling mode.
Synthetic-injection datasets generally produce higher scores, while real labeled datasets (especially highly imbalanced credit-card fraud data) are much harder and yield lower F1 across all methods.

5. For practical use, a hybrid strategy is recommended.
Use Isolation Forest as the primary detector, then use Z-Score/IQR for interpretable sanity checks and DBSCAN when density-based structure is expected.
