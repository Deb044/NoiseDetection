                                                        

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
                                                      
    img = plt.imread(path)

    if len(img.shape) == 3:
        img = np.mean(img, axis=2)

    img = img.astype(np.float32)

    if img.max() <= 1:
        img = img * 255.0

    return img


def iter_image_paths(folder: str | Path, limit: int | None = None) -> list[Path]:
                                                                
    folder_path = Path(folder)
    files = [
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ]
    files = sorted(files)
    return files if limit is None else files[:limit]
