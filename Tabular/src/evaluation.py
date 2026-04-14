"""Evaluation utilities for tabular noise detection outputs."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def summarize_labels(labels: np.ndarray, noise_label: int = -1) -> dict[str, float]:
    """Return noise count and percentage from a label vector."""
    total = int(labels.shape[0])
    noise_count = int(np.sum(labels == noise_label))
    noise_ratio = (noise_count / total) * 100 if total else 0.0

    return {
        "total_samples": float(total),
        "noise_count": float(noise_count),
        "noise_percentage": float(noise_ratio),
    }


def compare_detection_methods(method_to_indices: Mapping[str, list[int]], total_samples: int) -> dict[str, dict[str, float]]:
    """Compare methods by detected count and percentage."""
    if total_samples <= 0:
        raise ValueError("total_samples must be greater than 0")

    comparison: dict[str, dict[str, float]] = {}
    for method_name, indices in method_to_indices.items():
        detected_count = len(set(indices))
        comparison[method_name] = {
            "detected_count": float(detected_count),
            "detected_percentage": float((detected_count / total_samples) * 100),
        }

    return comparison
