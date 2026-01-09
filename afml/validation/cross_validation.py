"""Time-series cross-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


@dataclass
class PurgedKFold:
    """Purged K-Fold for time-series data."""

    n_splits: int
    embargo_pct: float = 0.0

    def split(self, X: pd.DataFrame, t1: pd.Series) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(X))
        test_starts = np.array_split(indices, self.n_splits)
        for test_idx in test_starts:
            test_start, test_end = test_idx[0], test_idx[-1]
            embargo = int(len(X) * self.embargo_pct)
            train_indices = np.concatenate([
                indices[:max(test_start - embargo, 0)],
                indices[min(test_end + embargo + 1, len(indices)) :],
            ])
            yield train_indices, test_idx


@dataclass
class CombinatorialPurgedKFold:
    """Combinatorial purged K-fold for time-series data."""

    n_splits: int
    n_test_splits: int
    embargo_pct: float = 0.0

    def split(self, X: pd.DataFrame, t1: pd.Series):
        indices = np.arange(len(X))
        splits = np.array_split(indices, self.n_splits)
        for i in range(self.n_splits - self.n_test_splits + 1):
            test_blocks = splits[i : i + self.n_test_splits]
            test_idx = np.concatenate(test_blocks)
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx


@dataclass
class WalkForwardAnalysis:
    """Walk-forward parameter search."""

    model_factory: callable
    param_grid: dict
    n_splits: int
    scoring: callable
    results_: list | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, t1: pd.Series, returns: pd.Series | None = None):
        grid = list(ParameterGrid(self.param_grid))
        split_points = np.array_split(np.arange(len(X)), self.n_splits)
        results = []
        for params in grid:
            scores = []
            for split in split_points:
                train_idx = np.setdiff1d(np.arange(len(X)), split)
                test_idx = split
                model = self.model_factory(**params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[test_idx])
                scores.append(self.scoring(y.iloc[test_idx], preds))
            results.append({"params": params, "score": np.mean(scores)})
        self.results_ = results
        return self

    def get_best_params(self) -> dict:
        if not self.results_:
            return {}
        best = max(self.results_, key=lambda item: item["score"])
        return best["params"]
