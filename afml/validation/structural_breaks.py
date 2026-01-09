"""Structural break detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class StructuralBreaks:
    """CUSUM-based structural break tests."""

    @staticmethod
    def get_cusum_stat(series: pd.Series, threshold: float = 5):
        s_pos, s_neg = 0.0, 0.0
        s_plus, s_minus = [], []
        for value in series.diff().fillna(0):
            s_pos = max(0.0, s_pos + value)
            s_neg = min(0.0, s_neg + value)
            s_plus.append(s_pos)
            s_minus.append(s_neg)
        stats = pd.DataFrame({"S+": s_plus, "S-": s_minus}, index=series.index)
        return stats, threshold

    @staticmethod
    def get_cusum_vol_stat(series: pd.Series, threshold: float = 10):
        vol = series.rolling(20).std().fillna(0)
        s_pos, s_neg = 0.0, 0.0
        s_plus, s_minus = [], []
        for value in vol.diff().fillna(0):
            s_pos = max(0.0, s_pos + value)
            s_neg = min(0.0, s_neg + value)
            s_plus.append(s_pos)
            s_minus.append(s_neg)
        stats = pd.DataFrame({"S+": s_plus, "S-": s_minus}, index=series.index)
        return stats, threshold
