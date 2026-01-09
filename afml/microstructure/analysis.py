"""Market microstructure analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class MarketMicrostructureAnalyzer:
    """Compute microstructure metrics."""

    def calculate_kyle_lambda(self, trades: pd.DataFrame, window: str = "1T"):
        signed_volume = trades["volume"] * np.sign(trades["price"].diff().fillna(0))
        grouped = trades.copy()
        grouped["signed_volume"] = signed_volume
        grouped = grouped.resample(window).sum()
        X = grouped[["signed_volume"]].fillna(0)
        y = grouped["price"].diff().fillna(0)
        model = LinearRegression().fit(X, y)
        lambda_val = float(model.coef_[0])
        return lambda_val, {"coef": lambda_val, "intercept": float(model.intercept_)}
