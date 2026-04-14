# DA213 Course Project - Python Programming Laboratory

This repository is a course project for DA213- Python Programming Laboratory.

## Team Members
- Debarghya Das - 240150049
- Maimoona Saifee - 240150043
- Ritwik Viswanathan - 240150031

## Project Overview
This project studies noise generation, detection, and handling across three data modalities:
- Image
- Tabular
- TimeSeries

Each modality follows a similar organization:
- notebooks: experiment notebooks
- src: reusable code modules
- report: modality report
- requirements.txt: modality-local dependency list

## Root Setup (All Modalities)
1. Clone the repository and move to project root.
2. Create a virtual environment.
3. Install root dependencies.
4. Launch Jupyter.

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

### Alternative (Jupyter Notebook)
```powershell
jupyter notebook
```

## Modality Setup and Usage

### 1. Tabular Mode
Path: Tabular

Run notebooks in sequence:
1. notebooks/01_iqr.ipynb
2. notebooks/02_zscore.ipynb
3. notebooks/03_dbscan.ipynb
4. notebooks/04_model_based.ipynb
5. notebooks/05_visualisation.ipynb

Reusable implementation is in:
- src/detectors.py
- src/evaluation.py
- src/visualization.py

Report:
- report/final_report.md

### 2. Image Mode
Path: Image

Run notebooks in sequence:
1. notebooks/01_data_loading_and_preview.ipynb
2. notebooks/02_noise_generation.ipynb
3. notebooks/03_basic_denoising_filters.ipynb
4. notebooks/04_adaptive_pipeline.ipynb
5. notebooks/05_medical_advanced_denoising.ipynb

Legacy reference notebook:
- notebooks/00_legacy_noise_handling.ipynb

Reusable implementation is in:
- src/io_utils.py
- src/noise_generation.py
- src/filters.py
- src/evaluation.py
- src/pipeline.py

Report:
- report/final_report.md

### 3. TimeSeries Mode
Path: TimeSeries

Run notebooks in sequence:
1. notebooks/01_data_and_noise.ipynb
2. notebooks/02_benchmarking.ipynb
3. notebooks/03_real_world_application.ipynb
4. notebooks/04_healthcare_application.ipynb

Reusable implementation is in:
- src/noise_generation.py
- src/filters.py
- src/evaluation.py

Report:
- report/Report.md

## Unified Project Structure
```text
NoiseDetection/
|-- README.md
|-- requirements.txt
|-- Image/
|   |-- images/
|   |   |-- medical/
|   |   |   |-- ultrasound/
|   |   |   `-- xray/
|   |   `-- random/
|   |-- notebooks/
|   |   |-- 00_legacy_noise_handling.ipynb
|   |   |-- 01_data_loading_and_preview.ipynb
|   |   |-- 02_noise_generation.ipynb
|   |   |-- 03_basic_denoising_filters.ipynb
|   |   |-- 04_adaptive_pipeline.ipynb
|   |   `-- 05_medical_advanced_denoising.ipynb
|   |-- report/
|   |   `-- final_report.md
|   |-- src/
|   |   |-- __init__.py
|   |   |-- evaluation.py
|   |   |-- filters.py
|   |   |-- io_utils.py
|   |   |-- noise_generation.py
|   |   `-- pipeline.py
|   `-- requirements.txt
|-- Tabular/
|   |-- notebooks/
|   |   |-- 01_iqr.ipynb
|   |   |-- 02_zscore.ipynb
|   |   |-- 03_dbscan.ipynb
|   |   |-- 04_model_based.ipynb
|   |   `-- 05_visualisation.ipynb
|   |-- report/
|   |   `-- final_report.md
|   |-- src/
|   |   |-- __init__.py
|   |   |-- detectors.py
|   |   |-- evaluation.py
|   |   `-- visualization.py
|   |-- requirements.txt
|   `-- ZScore.py
`-- TimeSeries/
    |-- notebooks/
    |   |-- 01_data_and_noise.ipynb
    |   |-- 02_benchmarking.ipynb
    |   |-- 03_real_world_application.ipynb
    |   `-- 04_healthcare_application.ipynb
    |-- report/
    |   `-- Report.md
    |-- src/
    |   |-- __init__.py
    |   |-- evaluation.py
    |   |-- filters.py
    |   |-- noise_generation.py
    |   `-- __pycache__/
    `-- requirements.txt
```

## Notes
- Use the root requirements.txt to run all modalities in one environment.