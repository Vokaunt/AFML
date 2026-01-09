"""Portfolio optimizer utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioOptimizer:
    """Simple optimizer for minimum variance and maximum Sharpe portfolios."""

    returns: pd.DataFrame
    risk_model: str = "sample"

    def _cov(self) -> pd.DataFrame:
        return self.returns.cov()

    def minimum_variance(self, max_position_size: float = 1.0) -> pd.Series:
        cov = self._cov()
        inv = np.linalg.pinv(cov.values)
        weights = inv.sum(axis=1)
        weights = weights / weights.sum()
        weights = np.clip(weights, -max_position_size, max_position_size)
        return pd.Series(weights, index=cov.index)

    def maximum_sharpe(self, max_position_size: float = 1.0) -> pd.Series:
        mean_returns = self.returns.mean()
        cov = self._cov()
        inv = np.linalg.pinv(cov.values)
        raw = inv @ mean_returns.values
        weights = raw / np.sum(np.abs(raw))
        weights = np.clip(weights, -max_position_size, max_position_size)
        return pd.Series(weights, index=cov.index)

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        mean_returns = self.returns.mean()
        cov = self._cov()
        weights = np.linspace(0, 1, n_points)
        points = []
        for w in weights:
            w_vec = np.full(len(mean_returns), w / len(mean_returns))
            ret = float(np.dot(w_vec, mean_returns.values))
            vol = float(np.sqrt(np.dot(w_vec.T, np.dot(cov.values, w_vec))))
            points.append({"return": ret, "volatility": vol})
        return pd.DataFrame(points)
