"""Bet sizing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone


@dataclass
class MetaLabeler:
    """Meta-labeling wrapper for primary and secondary models."""

    primary_model: object
    secondary_model: object

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.primary_model.fit(X, y)
        primary_pred = self.primary_model.predict(X)
        self.secondary_model.fit(X, primary_pred)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        primary_pred = self.primary_model.predict(X)
        return self.secondary_model.predict(pd.DataFrame(primary_pred, index=X.index))


class KellyBetSizing:
    """Kelly criterion sizing."""

    @staticmethod
    def size(prob: np.ndarray, win_loss_ratio: float = 1.0) -> np.ndarray:
        return (prob * (win_loss_ratio + 1) - 1) / win_loss_ratio


class BetSizingStrategies:
    """Collection of bet sizing heuristics."""

    @staticmethod
    def size_by_probability(probs: np.ndarray) -> np.ndarray:
        return (probs - 0.5) * 2

    @staticmethod
    def size_by_kelly(probs: np.ndarray, win_loss_ratio: float = 1.0) -> np.ndarray:
        return KellyBetSizing.size(probs, win_loss_ratio)
