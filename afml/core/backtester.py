"""Backtesting utilities used in AFML examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
import pandas as pd


@dataclass
class VectorizedBacktester:
    """Simple vectorized backtester for portfolio strategies."""

    prices: pd.DataFrame
    initial_capital: float = 1_000_000
    metrics: Dict[str, float] = field(default_factory=dict)
    weight_history: pd.DataFrame | None = None

    def run(
        self,
        strategy: Callable[[pd.DataFrame], pd.Series],
        rebalance_freq: str = "ME",
        lookback_periods: int | None = None,
    ) -> pd.DataFrame:
        """Run a rebalanced strategy and compute returns."""
        returns = self.prices.pct_change().dropna()
        weights = []
        rebalance_dates = returns.resample(rebalance_freq).last().index
        for date in rebalance_dates:
            window = returns.loc[:date]
            if lookback_periods is not None:
                window = window.iloc[-lookback_periods:]
            weight = strategy(window)
            weights.append(weight)
        weight_df = pd.DataFrame(weights, index=rebalance_dates).reindex(returns.index, method="ffill")
        self.weight_history = weight_df
        portfolio_returns = (weight_df * returns).sum(axis=1)
        results = pd.DataFrame({"returns": portfolio_returns})
        self._compute_metrics(portfolio_returns)
        return results

    def _compute_metrics(self, portfolio_returns: pd.Series) -> None:
        mean = portfolio_returns.mean()
        vol = portfolio_returns.std()
        sharpe = mean / vol * np.sqrt(252) if vol != 0 else np.nan
        self.metrics = {
            "mean_return": float(mean),
            "volatility": float(vol),
            "sharpe_ratio": float(sharpe),
        }


@dataclass
class StrategyBacktester:
    """Backtester for single series positions."""

    def backtest(self, returns: pd.Series, positions: pd.Series) -> Dict[str, float]:
        """Compute Sharpe ratio for a given strategy."""
        aligned_positions = positions.reindex(returns.index).fillna(0)
        pnl = returns * aligned_positions
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() != 0 else np.nan
        return {"sharpe_ratio": float(sharpe), "mean_return": float(pnl.mean())}
