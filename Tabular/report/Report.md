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
