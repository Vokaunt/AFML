"""Feature importance utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureImportance:
    """Feature importance measures."""

    @staticmethod
    def get_mdi_feature_importances(model, feature_names: list[str]) -> pd.Series:
        importances = pd.Series(model.feature_importances_, index=feature_names)
        return importances.sort_values(ascending=False)

    @staticmethod
    def get_mda_feature_importances(model, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> pd.Series:
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score

        baseline_scores = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            baseline_scores.append(accuracy_score(y.iloc[test_idx], preds))
        baseline = np.mean(baseline_scores)
        importances = {}
        for col in X.columns:
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            scores = []
            for train_idx, test_idx in kf.split(X_permuted):
                model.fit(X_permuted.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X_permuted.iloc[test_idx])
                scores.append(accuracy_score(y.iloc[test_idx], preds))
            importances[col] = baseline - np.mean(scores)
        return pd.Series(importances).sort_values(ascending=False)

    @staticmethod
    def plot_feature_importances(importances: pd.Series, title: str):
        return None

    @staticmethod
    def feature_importance_clustering(X: pd.DataFrame, importances: pd.Series, n_clusters: int = 5) -> pd.DataFrame:
        from sklearn.cluster import KMeans

        features = importances.loc[X.columns].fillna(0).values.reshape(-1, 1)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = model.fit_predict(features)
        return pd.DataFrame({"feature": X.columns, "cluster": clusters, "importance": importances.values})

    @staticmethod
    def select_features_from_clusters(cluster_df: pd.DataFrame) -> list[str]:
        return cluster_df.sort_values(["cluster", "importance"], ascending=[True, False]).groupby("cluster").first()[
            "feature"
        ].tolist()
