"""Fractional differentiation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FractionalDifferentiation:
    """Fractional differencing operations."""

    @staticmethod
    def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
        w = [1.0]
        k = 1
        while abs(w[-1]) >= thres:
            w.append(-w[-1] / k * (d - k + 1))
            k += 1
        w = np.array(w[::-1])
        width = len(w) - 1
        output = pd.Series(index=series.index)
        for idx in range(width, len(series)):
            window = series.iloc[idx - width : idx + 1]
            output.iloc[idx] = np.dot(w, window)
        return output.dropna()


class StationarityTests:
    """Stationarity tests for choosing d."""

    @staticmethod
    def find_optimal_d(series: pd.Series, d_values: np.ndarray | None = None):
        if d_values is None:
            d_values = np.linspace(0, 1, 11)
        results = []
        for d in d_values:
            diffed = FractionalDifferentiation.frac_diff_ffd(series, d)
            results.append({"d": d, "var": diffed.var()})
        results_df = pd.DataFrame(results)
        best = results_df.loc[results_df["var"].idxmin(), "d"]
        return best, results_df

    @staticmethod
    def plot_optimization_results(results_df: pd.DataFrame, optimal_d: float):
        return None
