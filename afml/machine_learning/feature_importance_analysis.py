"""Time-series and interaction feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_importance import FeatureImportance


class TimeSeriesFeatureImportance:
    """Rolling feature importance analysis."""

    @staticmethod
    def rolling_mean_decrease_accuracy(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int,
        step_size: int,
        test_size: int,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        results = []
        for start in range(0, len(X) - window_size - test_size + 1, step_size):
            train = slice(start, start + window_size)
            test = slice(start + window_size, start + window_size + test_size)
            model.fit(X.iloc[train], y.iloc[train])
            importances = FeatureImportance.get_mda_feature_importances(model, X.iloc[test], y.iloc[test])
            results.append(importances)
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results, index=range(len(results)))

    @staticmethod
    def plot_rolling_importance(rolling_importance: pd.DataFrame):
        return None


class FeatureInteractionImportance:
    """Pairwise interaction importance."""

    @staticmethod
    def pairwise_feature_importance(model, X: pd.DataFrame, y: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
        from sklearn.metrics import accuracy_score

        base_model = model.fit(X, y)
        base_preds = base_model.predict(X)
        base_score = accuracy_score(y, base_preds)
        interactions = []
        columns = list(X.columns)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                X_copy = X.copy()
                X_copy[columns[i]] = np.random.permutation(X_copy[columns[i]].values)
                X_copy[columns[j]] = np.random.permutation(X_copy[columns[j]].values)
                model.fit(X_copy, y)
                score = accuracy_score(y, model.predict(X_copy))
                interactions.append({"pair": (columns[i], columns[j]), "importance": base_score - score})
        return pd.DataFrame(interactions)

    @staticmethod
    def visualize_interaction_network(interaction_df: pd.DataFrame):
        return None
