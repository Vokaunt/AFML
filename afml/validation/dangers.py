"""Backtest danger detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class DangerDetector:
    """Detect potential backtest dangers like data leakage."""

    @staticmethod
    def detect_look_ahead_bias(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95) -> list[str]:
        suspicious = []
        for col in X.columns:
            corr = np.corrcoef(X[col].shift(-1).dropna(), y.loc[X.index[:-1]])[0, 1]
            if np.isnan(corr):
                continue
            if abs(corr) > threshold:
                suspicious.append(col)
        return suspicious

    @staticmethod
    def check_train_test_overlap(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
        overlap = X_train.index.intersection(X_test.index)
        return {
            "has_overlap": not overlap.empty,
            "overlap_count": len(overlap),
            "is_temporally_sound": X_train.index.max() < X_test.index.min(),
        }

    @staticmethod
    def detect_data_leakage(model, X_train, y_train, X_test, y_test) -> float:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        return score
