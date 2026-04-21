                                                                            

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_boxplot(series: pd.Series, title: str = "Boxplot for Noise Detection") -> None:
                                         
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=series, color="lightblue")
    plt.title(title, fontsize=14)
    plt.xlabel(series.name or "Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def plot_histogram_with_zscore(series: pd.Series, title: str = "Histogram with Z-Score Boundaries") -> None:
                                                                     
    values = series.astype(float)
    mean = values.mean()
    std = values.std()

    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=40, color="gray", edgecolor="black", alpha=0.7)
    plt.axvline(mean, color="blue", linestyle="--", label=f"Mean: {mean:.2f}")

    for z in [-3, -2, 2, 3]:
        color = "green" if abs(z) == 2 else "red"
        label = f"{z} SD" if abs(z) == 3 else None
        plt.axvline(mean + z * std, color=color, linestyle="--", label=label)

    plt.title(title, fontsize=14)
    plt.xlabel(series.name or "Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()


def plot_pairplot(df: pd.DataFrame, title: str = "Multivariate Pairplot for Noise Detection") -> None:
                                                                        
    grid = sns.pairplot(df, diag_kind="kde", corner=True, plot_kws={"color": "darkblue", "alpha": 0.6})
    grid.figure.suptitle(title, fontsize=16, y=1.02)
    plt.show()
