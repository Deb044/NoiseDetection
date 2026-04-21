                                                                

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_outliers_iqr(data: Iterable[float], multiplier: float = 1.5) -> tuple[list[int], tuple[float, float]]:
                                                    
    values = np.asarray(list(data), dtype=float)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = np.where((values < lower_bound) | (values > upper_bound))[0].tolist()
    return outliers, (float(lower_bound), float(upper_bound))


def detect_outliers_zscore(data: Iterable[float], threshold: float = 3.0) -> tuple[list[int], list[float]]:
                                                               
    values = np.asarray(list(data), dtype=float)
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return [], [0.0 for _ in values]

    z_scores = ((values - mean) / std).tolist()
    outliers = [idx for idx, score in enumerate(z_scores) if abs(score) > threshold]
    return outliers, z_scores


def detect_noise_dbscan(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    scale: bool = True,
) -> np.ndarray:
                                                                
    input_df = df.copy()
    if scale:
        input_df = pd.DataFrame(StandardScaler().fit_transform(input_df), columns=df.columns)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(input_df)


def detect_noise_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
                                                                     
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(df)
    scores = model.decision_function(df)
    return labels, scores
