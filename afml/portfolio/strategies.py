"""Portfolio strategy implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_strategy(returns: pd.DataFrame) -> pd.Series:
    n_assets = returns.shape[1]
    return pd.Series(1 / n_assets, index=returns.columns)


def minimum_variance_strategy(returns: pd.DataFrame) -> pd.Series:
    cov = returns.cov()
    inv = np.linalg.pinv(cov.values)
    weights = inv.sum(axis=1)
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns)


def momentum_strategy(returns: pd.DataFrame, lookback: int = 20) -> pd.Series:
    scores = returns.tail(lookback).mean()
    weights = scores / scores.abs().sum()
    return weights.fillna(0)
