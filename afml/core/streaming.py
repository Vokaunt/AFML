"""Streaming and Kalman filter backtesters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class StreamingBacktester:
    """Chunked backtester for streaming-style evaluation."""

    data_loader: Callable[[int, int], pd.DataFrame]
    total_rows: int
    feature_cols: List[str]
    target_col: str
    chunk_size: int = 100

    def run(
        self,
        model_factory: Callable[[], Ridge],
        initial_train_size: int,
        retrain_freq: int,
    ) -> Dict[str, float]:
        """Iteratively train and evaluate on chunks of data."""
        model = model_factory()
        preds = []
        actuals = []
        for start in range(initial_train_size, self.total_rows, self.chunk_size):
            end = min(start + self.chunk_size, self.total_rows)
            train = self.data_loader(0, start)
            test = self.data_loader(start, end)
            if start == initial_train_size or start % retrain_freq == 0:
                model = model_factory()
                model.fit(train[self.feature_cols], train[self.target_col])
            preds.append(model.predict(test[self.feature_cols]))
            actuals.append(test[self.target_col].values)
        preds = np.concatenate(preds) if preds else np.array([])
        actuals = np.concatenate(actuals) if actuals else np.array([])
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2))) if preds.size else float("nan")
        return {"rmse": rmse}


@dataclass
class KalmanFilterBacktester:
    """Simple Kalman filter regression backtester."""

    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Run a basic Kalman filter on one feature."""
        x = X.iloc[:, 0].values
        n = len(x)
        beta = 0.0
        P = 1.0
        Q = 1e-5
        R = 1e-2
        preds = []
        for i in range(n):
            P = P + Q
            K = P / (P + R)
            beta = beta + K * (y.iloc[i] - beta * x[i])
            P = (1 - K) * P
            preds.append(beta * x[i])
        preds = np.array(preds)
        rmse = float(np.sqrt(np.mean((preds - y.values) ** 2)))
        return {"rmse": rmse}
