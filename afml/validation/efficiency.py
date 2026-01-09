"""Efficiency analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


@dataclass
class EfficiencyAnalyzer:
    """Analyze approximation errors via subsampling."""

    def analyze_approximation_error(
        self,
        model_factory: Callable[[], object],
        X: pd.DataFrame,
        y: pd.Series,
        subsample_ratios: list[float],
        n_repetitions: int,
        cv_splits: int,
    ) -> dict:
        results = []
        for ratio in subsample_ratios:
            scores = []
            for _ in range(n_repetitions):
                sample_size = int(len(X) * ratio)
                sample_idx = np.random.choice(len(X), size=sample_size, replace=False)
                X_sub = X.iloc[sample_idx]
                y_sub = y.iloc[sample_idx]
                kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
                for train_idx, test_idx in kf.split(X_sub):
                    model = model_factory()
                    model.fit(X_sub.iloc[train_idx], y_sub.iloc[train_idx])
                    preds = model.predict(X_sub.iloc[test_idx])
                    scores.append(np.mean((preds - y_sub.iloc[test_idx]) ** 2))
            results.append({"ratio": ratio, "mse": np.mean(scores)})
        return {"results_df": pd.DataFrame(results)}
