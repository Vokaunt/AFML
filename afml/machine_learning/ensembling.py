"""Ensembling strategies for AFML examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


@dataclass
class DisjointFeatureEnsemble:
    base_estimator: object
    n_estimators: int
    random_state: int | None = None
    models: List[object] = None
    feature_splits: List[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        rng = np.random.default_rng(self.random_state)
        features = list(X.columns)
        rng.shuffle(features)
        splits = np.array_split(features, self.n_estimators)
        self.feature_splits = [list(split) for split in splits]
        self.models = []
        for split in self.feature_splits:
            model = clone(self.base_estimator)
            model.fit(X[split], y)
            self.models.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for model, split in zip(self.models, self.feature_splits):
            preds.append(model.predict(X[split]))
        return np.round(np.mean(preds, axis=0)).astype(int)


@dataclass
class DiversityEnsemble:
    base_estimator: object
    n_estimators: int
    random_state: int | None = None
    models: List[object] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        rng = np.random.default_rng(self.random_state)
        self.models = []
        for _ in range(self.n_estimators):
            idx = rng.choice(len(X), size=len(X), replace=True)
            model = clone(self.base_estimator)
            model.fit(X.iloc[idx], y.iloc[idx])
            self.models.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.array([model.predict(X) for model in self.models])
        return np.round(preds.mean(axis=0)).astype(int)


@dataclass
class StackedGeneralizationEnsemble:
    base_estimators: List[object]
    meta_estimator: object
    cv: int = 3
    base_models: List[object] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_estimators)))
        self.base_models = []
        for i, estimator in enumerate(self.base_estimators):
            fold_preds = np.zeros(len(X))
            for train_idx, test_idx in kf.split(X):
                model = clone(estimator)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                fold_preds[test_idx] = model.predict(X.iloc[test_idx])
            meta_features[:, i] = fold_preds
            self.base_models.append(clone(estimator).fit(X, y))
        self.meta_estimator.fit(meta_features, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = np.column_stack([model.predict(X) for model in self.base_models])
        return self.meta_estimator.predict(meta_features)


@dataclass
class BetSizingEnsemble:
    base_estimators: List[object]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for estimator in self.base_estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.array([estimator.predict(X) for estimator in self.base_estimators])
        return np.round(preds.mean(axis=0)).astype(int)

    def predict_bet_size(self, X: pd.DataFrame) -> np.ndarray:
        probs = []
        for estimator in self.base_estimators:
            if hasattr(estimator, "predict_proba"):
                probs.append(estimator.predict_proba(X)[:, 1])
            else:
                probs.append(estimator.predict(X))
        return np.mean(probs, axis=0)
