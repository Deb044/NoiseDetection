# Deep Dive Analysis: Time Series Noise Diagnostics and Filtering

## Overview
This project presents a deep-dive analysis into time series noise diagnostics and filtering techniques. The primary objective is to define various shapes of time series noise explicitly, develop custom from-scratch algorithms for noise isolation and removal, and benchmark their efficacy. By mathematically proving the downstream performance improvements of rigorous signal filtering, this project underscores the criticality of preprocessing in applications ranging from IoT telemetry to medical monitoring.

## System Architecture / Workflow
1. **Signal Generation:** Creation of pure synthetic signals composed of multiple frequency components.
2. **Noise Injection:** Systematic simulation of real-world noise archetypes (e.g., Gaussian, Spike, Drift, Seasonal, and Missing Data).
3. **Filtering Pipeline:** Application of various custom-built filtering algorithms (Moving Average, Median Filter, STL Decomposition, Kalman Filter, Fourier Low-Pass) to recover the ground truth.
4. **Evaluation:** Benchmarking filter performance using robust statistical metrics (RMSE, MAE, SNR) across different data configurations.
5. **Real-World Applications:** Validating methodologies on actual datasets like the Statsmodels Sunspots Dataset and Scipy ECG datasets.

## Methodology / Approach Used
The methodology involves an empirical analysis comparing algorithmic resilience against specific noise classes:
*   **High-Frequency Noise Resilience:** Analyzing Gaussian and Spike noise impact, and mitigating through Moving Averages or Median filtering.
*   **Low-Frequency Artifact Correction:** Correcting Sensor Drift and Seasonal Interference through detrending, classical STL-style decompositions, and Low-Pass Fourier transforms.
*   **Data Integrity Restoration:** Healing missing values and NaN blocks via linear interpolation. 
*   **Comparative Analysis:** All evaluations compare the original "clean" proxy against the filtered result based on multiple evaluation matrices over identical window periods.

## Dataset Details
* **Synthetic Signals:** Generated procedurally using trigonometric components (e.g., sine waves with 1Hz, 0.2Hz, 5.0Hz frequency mixes).
* **Sunspots Dataset (`statsmodels`):** A real-world dataset detailing historical sunspot activity, utilized to observe the effect of missingness and spikes out-of-sample.
* **ECG Telemetry Dataset (`scipy.datasets`):** Real electrocardiogram data representing complex overlapping noise (respiratory baseline wander, 50Hz AC powerline hum, muscle twitching).

## Implementation Details
The project is built entirely from scratch in Python, utilizing `numpy` for array convolutions and fast mathematical operations.

**Algorithms Implemented (`src/filters.py`, `src/noise_generation.py`, `src/evaluation.py`):**
*   **Filters:**
    *   *Moving Average Filter:* A basic convolution filter for rapid smoothing.
    *   *Median Filter:* Robust non-linear outlier replacement, highly effective for spike noise.
    *   *Classical STL-like Decomposition:* Manual additive separation of trend, seasonality, and residuals.
    *   *1D Kalman Filter:* Recursive optimal estimation balancing process variance and measurement variance.
    *   *Fourier Low-Pass Filter (FFT):* Frequency domain projection to completely zero-out components above a critical threshold.
    *   *Linear Interpolation:* Gap-filling algorithm to traverse NaNs safely.
*   **Noise Generators:** Functions to selectively add Gaussian, Spike, Random Walk Drift, Linear Drift, Seasonal interference, and Dropouts.
*   **Evaluation Metrics:** Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Signal-to-Noise Ratio (SNR).
*   **External Tools/Models for downstream task validation:** `XGBoost` / `GradientBoostingRegressor` to demonstrate forecast improvements on cleaned datasets.

## How to Run the Project
*Prerequisites:* Ensure you have Python installed with the requirements listed in `requirements.txt`.

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd TimeSeries
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute the Jupyter Notebooks sequentially** from the `notebooks/` directory to reproduce the experiments:
   *   `01_data_and_noise.ipynb`: Explore synthetic generation and noise profiles.
   *   `02_benchmarking.ipynb`: Run the cross-matrix benchmarking of different algorithms vs. noise.
   *   `03_real_world_application.ipynb`: Validate uplift using the real-world Sunspots dataset.
   *   `04_healthcare_application.ipynb`: Perform complex dual-pipeline ECG restitution.
4. *Optional:* The source implementations can be directly imported from the `src/` modules into your own scripts.

## Results / Outputs
*   **Gaussian Noise:** Best mitigated by the **Kalman Filter** or **Fourier LPF**.
*   **Spike Noise:** Severely distorted convolutions but was nearly perfectly neutralized by the **Median Filter**.
*   **Seasonal Interference:** Effectively stripped via **STL Decomposition** or targeted FFT filtering.
*   **Missing Blocks:** Instantaneously break normal algorithms running pure FFT and need prior **Linear Interpolation**.
*   **Predictive Model Uplift:** On the degraded Sunspot dataset, our Median Filter recovered the majority of predictive accuracy. 
    *   *RMSE Clean Ground Truth:* ~12.21
    *   *RMSE Damaged Data:* ~19.34
    *   *RMSE Filtered Recovery:* ~13.65
*   **ECG Healthcare Pipeline:** A dual Moving Average (High-Pass trace) + Fourier Low-Pass pipeline effectively wiped baseline wander and powerline hum, restoring >95% of original QRS waveform integrity.

*(Note: Please refer to the respective notebooks for visual outputs, plots, and screenshots charting each filtering stage.)*

## Limitations and Future Scope
*   **Limitations:** The current filters are predominantly univariate. The customized STL-style decomposition performs well with fixed integer periods but may struggle with variable-frequency seasonality. Simple Kalman implementations assume mostly static process variances and may underperform under highly volatile non-stationary environments.
*   **Future Scope:** 
    * Extending architectures to handle multivariate telemetry arrays (sensor fusion).
    * Incorporating adaptive filtering frameworks (e.g., Extended Kalman Filters or LMS) for non-stationary drifts.
    * Incorporating deep-learning paradigms like Autoencoders specialized in mapping noisy to clean signal profiles for complex non-linear feature structures.

## References
* Statsmodels Datasets: `statsmodels.api.datasets`
* SciPy Library and Signal/Dataset Modules
* Time Series Analysis and Classical Filtering Theory
