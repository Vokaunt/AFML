"""Backtest robustness checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


class RobustnessChecker:
    """Robustness tests for model performance."""

    @staticmethod
    def walk_forward_validation(
        model_class,
        params: dict,
        X: pd.DataFrame,
        y: pd.Series,
        initial_train_size: int,
        step_size: int,
        metric,
    ) -> dict:
        test_scores = []
        for start in range(initial_train_size, len(X), step_size):
            train_X = X.iloc[:start]
            train_y = y.iloc[:start]
            test_X = X.iloc[start : start + step_size]
            test_y = y.iloc[start : start + step_size]
            model = model_class(**params)
            model.fit(train_X, train_y)
            preds = model.predict(test_X)
            test_scores.append(metric(test_y, preds))
        consistency = float(np.std(test_scores)) if test_scores else float("nan")
        return {"test_scores": test_scores, "consistency": consistency}

    @staticmethod
    def subsample_robustness_test(
        model_class,
        params: dict,
        X: pd.DataFrame,
        y: pd.Series,
        n_iterations: int,
        sample_fraction: float,
        metric,
    ) -> dict:
        scores = []
        for _ in range(n_iterations):
            sample_idx = np.random.choice(len(X), size=int(len(X) * sample_fraction), replace=False)
            X_sub = X.iloc[sample_idx]
            y_sub = y.iloc[sample_idx]
            model = model_class(**params)
            model.fit(X_sub, y_sub)
            preds = model.predict(X_sub)
            scores.append(metric(y_sub, preds))
        stability = float(np.std(scores)) if scores else float("nan")
        return {"test_scores": scores, "stability": stability}

    @staticmethod
    def parameter_stability_test(model_class, param_grid: dict, X: pd.DataFrame, y: pd.Series, n_periods: int = 4):
        grid = list(ParameterGrid(param_grid))
        period_length = len(X) // n_periods
        results = []
        for params in grid:
            scores = []
            for i in range(n_periods):
                start = i * period_length
                end = (i + 1) * period_length if i < n_periods - 1 else len(X)
                model = model_class(**params)
                model.fit(X.iloc[start:end], y.iloc[start:end])
                scores.append(model.score(X.iloc[start:end], y.iloc[start:end]))
            results.append({"params": params, "scores": scores})
        return results
