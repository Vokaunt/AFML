"""Synthetic data generators for AFML examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticData:
    """Generate synthetic price series for demonstrations."""

    seed: int | None = None

    def generate_prices(self, n_assets: int, n_days: int) -> pd.DataFrame:
        """Generate geometric random walk prices."""
        rng = np.random.default_rng(self.seed)
        rets = rng.normal(0, 0.01, size=(n_days, n_assets))
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
        return pd.DataFrame(prices, index=dates, columns=[f"asset_{i}" for i in range(n_assets)])


class BacktestSimulator:
    """Simulation utilities for backtest diagnostics."""

    @staticmethod
    def demonstrate_backtest_overfitting(
        n_strategies: int,
        n_periods: int,
        is_fraction: float,
        random_state: int | None = None,
    ) -> dict:
        """Simulate strategies and compute a simple probability of backtest overfitting."""
        rng = np.random.default_rng(random_state)
        returns = rng.normal(0, 0.01, size=(n_periods, n_strategies))
        split = int(n_periods * is_fraction)
        is_scores = returns[:split].mean(axis=0)
        oos_scores = returns[split:].mean(axis=0)
        best_is = np.argmax(is_scores)
        pbo = float(oos_scores[best_is] < np.median(oos_scores))
        dsr = float(np.mean(oos_scores))
        return {
            "pbo": pbo,
            "dsr": dsr,
            "pbo_fig": None,
            "degradation_fig": None,
        }
