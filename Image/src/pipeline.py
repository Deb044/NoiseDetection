                                                                        

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import psnr
from .filters import denoise_tv, gaussian_filter, median_filter, remove_speckle, wavelet_denoise
from .io_utils import iter_image_paths, load_image
from .noise_generation import (
    add_gaussian,
    add_mixed,
    add_poisson,
    add_salt_pepper,
    add_speckle,
    add_uniform,
    detect_noise,
)


def process_image(img: np.ndarray, noise_type: str | None = None) -> tuple[str, str, np.ndarray, np.ndarray]:
                                                                    
    if noise_type is None:
        noise_type = np.random.choice(["gaussian", "salt_pepper", "speckle", "poisson", "uniform", "mixed"])

    if noise_type == "gaussian":
        noisy = add_gaussian(img)
    elif noise_type == "salt_pepper":
        noisy = add_salt_pepper(img)
    elif noise_type == "speckle":
        noisy = add_speckle(img)
    elif noise_type == "poisson":
        noisy = add_poisson(img)
    elif noise_type == "uniform":
        noisy = add_uniform(img)
    else:
        noisy = add_mixed(img)

    detected = detect_noise(noisy)

    if detected == "salt_pepper":
        clean = median_filter(noisy)
    elif detected == "gaussian":
        clean = gaussian_filter(noisy)
    else:
        clean = remove_speckle(noisy)

    return noise_type, detected, noisy, clean


def show_triplet(original: np.ndarray, noisy: np.ndarray, clean: np.ndarray, noise_type: str, detected: str) -> None:
                                                                 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")

    axes[1].imshow(noisy, cmap="gray")
    axes[1].set_title(f"Noisy\\n({noise_type})")

    axes[2].imshow(clean, cmap="gray")
    axes[2].set_title(f"Denoised\\nDetected: {detected}")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"PSNR: {psnr(original, clean):.2f}", fontsize=12)
    plt.tight_layout()
    plt.show()


def run_basic_folder(folder: str | Path, noise_type: str | None = None, limit: int = 5) -> None:
                                                               
    for path in iter_image_paths(folder, limit):
        original = load_image(path)
        used_noise, detected, noisy, clean = process_image(original, noise_type)
        print(f"File: {path.name} | Added: {used_noise} | Detected: {detected}")
        show_triplet(original, noisy, clean, used_noise, detected)


def run_medical_comparison(folder: str | Path, limit: int = 4) -> None:
                                                                      
    for path in iter_image_paths(folder, limit):
        original = load_image(path)
        noisy = add_gaussian(original, sigma=20)

        clean_tv = denoise_tv(noisy, weight=50.0, iterations=120)
        clean_wv = wavelet_denoise(noisy, threshold=90, levels=2)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        images = [original, noisy, clean_tv, clean_wv]
        titles = ["Original", "Heavy Noise", "TV (Iterative)", "Wavelet (Multi-level)"]

        for i, ax in enumerate(axes):
            low, high = np.percentile(images[i], (2, 98))
            ax.imshow(images[i], cmap="gray", vmin=low, vmax=high)
            if i > 1:
                ax.set_title(f"{titles[i]}\\nPSNR: {psnr(original, images[i]):.2f}")
            else:
                ax.set_title(titles[i])
            ax.axis("off")

        fig.suptitle(f"File: {path.name}", fontsize=12)
        plt.tight_layout()
        plt.show()
