"""Performance statistics utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PerformanceStatistics:
    """Common backtest statistics."""

    @staticmethod
    def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
        mean = returns.mean()
        vol = returns.std()
        return float(mean / vol * np.sqrt(annualization)) if vol != 0 else float("nan")

    @staticmethod
    def drawdown(returns: pd.Series) -> pd.Series:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        return (cumulative - peak) / peak
