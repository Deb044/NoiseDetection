# Final Report: Image Noise Handling

## 1. Objective
Build a reusable and readable image-noise workflow that supports:

- synthetic noise generation,
- adaptive denoising,
- medical-image focused denoising comparisons,
- quantitative evaluation using PSNR.

## 2. Folder Structure

- notebooks: modular experiment notebooks
- src: reusable code modules
- report: final documentation
- requirements.txt: dependency list

## 3. Notebooks

1. notebooks/01_data_loading_and_preview.ipynb
2. notebooks/02_noise_generation.ipynb
3. notebooks/03_basic_denoising_filters.ipynb
4. notebooks/04_adaptive_pipeline.ipynb
5. notebooks/05_medical_advanced_denoising.ipynb

Legacy reference:
- notebooks/00_legacy_noise_handling.ipynb

## 4. Source Modules

- src/io_utils.py: image loading and folder iteration
- src/noise_generation.py: noise injection and noise heuristics
- src/filters.py: median/gaussian/log-domain filters, TV denoise, wavelet denoise
- src/evaluation.py: MSE and PSNR metrics
- src/pipeline.py: end-to-end processing and visualization runners
