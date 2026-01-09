"""Portfolio analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioAnalytics:
    """Portfolio analytics for performance and risk."""

    returns: pd.DataFrame
    weights: pd.DataFrame

    def performance_summary(self) -> dict:
        portfolio_returns = (self.weights * self.returns).sum(axis=1)
        total_return = (1 + portfolio_returns).prod() - 1
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
        }

    def calculate_risk_contributions(self, lookback: int = 252) -> pd.DataFrame:
        rolling_cov = self.returns.rolling(lookback).cov()
        contributions = []
        for date in self.weights.index:
            cov = rolling_cov.loc[date]
            if cov.isnull().values.any():
                continue
            w = self.weights.loc[date].values.reshape(-1, 1)
            total_var = float(w.T @ cov.values @ w)
            contrib = (w * (cov.values @ w)) / total_var
            contributions.append(pd.Series(contrib.flatten(), index=self.weights.columns, name=date))
        return pd.DataFrame(contributions)

    @staticmethod
    def calculate_risk_concentration(risk_contributions: pd.DataFrame) -> pd.Series:
        return (risk_contributions ** 2).sum(axis=1)
