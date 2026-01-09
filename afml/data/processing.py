"""Multi-product series processing helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def etf_trick(price_dict: Dict[str, pd.DataFrame], volume_dict: Dict[str, pd.Series]) -> pd.Series:
    """Construct a continuous series from multiple ETFs via volume weighting."""
    weights = []
    prices = []
    for key, df in price_dict.items():
        vol = volume_dict[key].reindex(df.index).fillna(method="ffill")
        weights.append(vol)
        prices.append(df.iloc[:, 0])
    weights_df = pd.concat(weights, axis=1).fillna(0)
    price_df = pd.concat(prices, axis=1).fillna(method="ffill")
    weight_sum = weights_df.sum(axis=1).replace(0, np.nan)
    return (price_df.mul(weights_df).sum(axis=1) / weight_sum).dropna()


def futures_roll(front: pd.Series, back: pd.Series, roll_date: pd.Timestamp) -> pd.Series:
    """Roll a futures series at a given date."""
    combined = front.copy()
    combined.loc[roll_date:] = back.loc[roll_date:]
    return combined


def pca_hedge_weights(cov: pd.DataFrame, n_components: int = 1) -> pd.Series:
    """Compute hedge weights using principal component analysis."""
    eigvals, eigvecs = np.linalg.eigh(cov.values)
    idx = np.argsort(eigvals)[::-1]
    principal = eigvecs[:, idx[:n_components]]
    weights = principal.mean(axis=1)
    weights = weights / np.sum(np.abs(weights))
    return pd.Series(weights, index=cov.index)
