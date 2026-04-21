                                                          

from __future__ import annotations

import numpy as np


def add_gaussian(img: np.ndarray, sigma: float = 30) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255)


def add_salt_pepper(img: np.ndarray, prob: float = 0.03) -> np.ndarray:
    noisy = img.copy()
    rand = np.random.rand(*img.shape)
    noisy[rand < prob / 2] = 0
    noisy[rand > 1 - prob / 2] = 255
    return noisy


def add_speckle(img: np.ndarray, sigma: float = 0.4) -> np.ndarray:
    noise = np.random.randn(*img.shape)
    return np.clip(img + img * noise * sigma, 0, 255)


def add_poisson(img: np.ndarray) -> np.ndarray:
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(max(vals, 1)))
    noisy = np.random.poisson(img * vals) / vals
    return np.clip(noisy, 0, 255)


def add_uniform(img: np.ndarray, low: float = -30, high: float = 30) -> np.ndarray:
    noise = np.random.uniform(low, high, img.shape)
    return np.clip(img + noise, 0, 255)


def add_mixed(img: np.ndarray) -> np.ndarray:
    mixed = add_gaussian(img, 15)
    mixed = add_salt_pepper(mixed, 0.03)
    return mixed


def detect_noise(img: np.ndarray) -> str:
                                                                      
    total = img.size
    extreme = np.sum((img <= 5) | (img >= 250))
    ratio = extreme / total
    var = np.var(img)

    if var > 1500:
        return "gaussian"
    if var > 500:
        return "speckle"
    if ratio > 0.02:
        return "salt_pepper"
    return "uniform"
