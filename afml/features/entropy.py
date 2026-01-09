"""Entropy-based features."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


class EntropyFeatures:
    """Entropy-based feature engineering."""

    @staticmethod
    def discretize_series(series: pd.Series, n_bins: int = 10) -> pd.Series:
        return pd.qcut(series, n_bins, labels=False, duplicates="drop")

    @staticmethod
    def plug_in_entropy(series: pd.Series) -> float:
        counts = series.value_counts(normalize=True)
        return float(-(counts * np.log2(counts)).sum())

    @staticmethod
    def lempel_ziv_complexity(sequence: str) -> float:
        i, k, l = 0, 1, 1
        complexity = 1
        while True:
            if i + k == len(sequence):
                complexity += 1
                break
            if sequence[i + k] == sequence[l + k - 1]:
                k += 1
                if l + k > len(sequence):
                    complexity += 1
                    break
            else:
                if k > 1:
                    i += 1
                    k = 1
                else:
                    complexity += 1
                    l += 1
                    i = 0
                    if l == len(sequence):
                        break
        return complexity / len(sequence)

    @staticmethod
    def rolling_entropy(series: pd.Series, window: int, method: str = "plug_in") -> pd.Series:
        values = []
        for i in range(window, len(series) + 1):
            window_slice = series.iloc[i - window : i]
            if method == "plug_in":
                discretized = EntropyFeatures.discretize_series(window_slice)
                values.append(EntropyFeatures.plug_in_entropy(discretized))
            elif method == "lz":
                binary_seq = "".join((window_slice > window_slice.mean()).astype(int).astype(str))
                values.append(EntropyFeatures.lempel_ziv_complexity(binary_seq))
            else:
                raise ValueError("Unknown entropy method")
        return pd.Series(values, index=series.index[window - 1 :])
