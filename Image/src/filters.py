"""Denoising filters for natural and medical images."""

from __future__ import annotations

import numpy as np


def median_filter(img: np.ndarray, k: int = 3) -> np.ndarray:
    pad = k // 2
    padded = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i : i + k, j : j + k]
            out[i, j] = np.median(window)
    return out


def gaussian_kernel(k: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(k // 2), k // 2, k)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def gaussian_filter(img: np.ndarray, k: int = 3, sigma: float = 1.0) -> np.ndarray:
    kernel = gaussian_kernel(k, sigma)
    pad = k // 2
    padded = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i : i + k, j : j + k]
            out[i, j] = np.sum(window * kernel)
    return out


def remove_speckle(img: np.ndarray) -> np.ndarray:
    safe = img + 1e-5
    log_img = np.log(safe)
    filtered = gaussian_filter(log_img, 3, 1)
    return np.exp(filtered)


def denoise_tv(img: np.ndarray, weight: float = 40.0, iterations: int = 1000) -> np.ndarray:
    """Simple iterative TV-like denoising update."""
    u = img.copy()
    f = img.copy()
    dt = 0.125

    for _ in range(iterations):
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        mag = np.sqrt(ux**2 + uy**2 + 0.01)
        nx = ux / mag
        ny = uy / mag
        div = (nx - np.roll(nx, 1, axis=1)) + (ny - np.roll(ny, 1, axis=0))

        u = u + dt * (f - u + weight * div)

    return np.clip(u, 0, 255)


def haar_dwt_2d(img: np.ndarray) -> np.ndarray:
    l = (img[:, 0::2] + img[:, 1::2]) / np.sqrt(2)
    h = (img[:, 0::2] - img[:, 1::2]) / np.sqrt(2)
    row_trans = np.hstack((l, h))

    top = (row_trans[0::2, :] + row_trans[1::2, :]) / np.sqrt(2)
    bottom = (row_trans[0::2, :] - row_trans[1::2, :]) / np.sqrt(2)
    return np.vstack((top, bottom))


def haar_idwt_2d(coeffs: np.ndarray) -> np.ndarray:
    h, w = coeffs.shape
    h2, w2 = h // 2, w // 2

    top = coeffs[:h2, :]
    bottom = coeffs[h2:, :]

    col_inv = np.zeros((h, w))
    col_inv[0::2, :] = (top + bottom) / np.sqrt(2)
    col_inv[1::2, :] = (top - bottom) / np.sqrt(2)

    left = col_inv[:, :w2]
    right = col_inv[:, w2:]

    final = np.zeros((h, w))
    final[:, 0::2] = (left + right) / np.sqrt(2)
    final[:, 1::2] = (left - right) / np.sqrt(2)
    return final


def wavelet_denoise(img: np.ndarray, threshold: float = 80, levels: int = 2) -> np.ndarray:
    output = img.copy()

    for _ in range(levels):
        h, w = output.shape
        work_img = output[: h // 2 * 2, : w // 2 * 2]
        coeffs = haar_dwt_2d(work_img)

        denoised_coeffs = np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
        h2, w2 = coeffs.shape[0] // 2, coeffs.shape[1] // 2
        denoised_coeffs[:h2, :w2] = coeffs[:h2, :w2]

        output = haar_idwt_2d(denoised_coeffs)

    return np.clip(output, 0, 255)
