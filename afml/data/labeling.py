"""Labeling utilities for AFML examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DailyVolatility:
    """Daily volatility estimators."""

    @staticmethod
    def get_daily_vol(prices: pd.Series, span: int = 30) -> pd.Series:
        returns = prices.pct_change().dropna()
        return returns.ewm(span=span).std()


@dataclass
class TripleBarrierLabeling:
    """Triple barrier labeling implementation."""

    prices: pd.Series
    events: pd.DataFrame
    pt_sl: tuple[float, float]
    min_ret: float
    num_threads: int = 1

    def get_events(self) -> pd.DataFrame:
        events = self.events.copy()
        pt, sl = self.pt_sl
        out = events.copy()
        out["t1"] = events["tl"]
        out["trgt"] = events["trgt"]
        out = out[out["trgt"] > self.min_ret]
        for idx in out.index:
            price_path = self.prices.loc[idx:out.loc[idx, "t1"]]
            returns = price_path / price_path.iloc[0] - 1
            if pt > 0:
                touched = returns[returns > pt * out.loc[idx, "trgt"]]
                if not touched.empty:
                    out.loc[idx, "t1"] = touched.index[0]
            if sl > 0:
                touched = returns[returns < -sl * out.loc[idx, "trgt"]]
                if not touched.empty:
                    out.loc[idx, "t1"] = min(out.loc[idx, "t1"], touched.index[0])
        return out

    @staticmethod
    def get_bins(events: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        aligned = events.dropna(subset=["t1"]).copy()
        px = prices.reindex(aligned.index.union(aligned["t1"]).unique()).ffill()
        returns = px.loc[aligned["t1"].values].values / px.loc[aligned.index] - 1
        out = pd.DataFrame(index=aligned.index)
        out["ret"] = returns
        out["bin"] = np.sign(returns)
        return out


class MetaLabeling:
    """Meta-labeling helpers."""

    @staticmethod
    def bet_size(prob: float) -> float:
        return (prob - 0.5) * 2

    @staticmethod
    def plot_precision_vs_accuracy(probs: np.ndarray, labels: np.ndarray):
        # Placeholder for demonstration
        return None
