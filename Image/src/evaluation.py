                                               

from __future__ import annotations

import numpy as np


def mse(original: np.ndarray, denoised: np.ndarray) -> float:
    return float(np.mean((original - denoised) ** 2))


def psnr(original: np.ndarray, denoised: np.ndarray) -> float:
    error = mse(original, denoised)
    if error == 0:
        return float("inf")
    return float(20 * np.log10(255 / np.sqrt(error)))
