"""Sampling utilities used in AFML examples."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SequentialBootstrap:
    """Sequential bootstrap sampling."""

    @staticmethod
    def get_ind_matrix(index: pd.Index, embargo_time: pd.Timedelta) -> pd.DataFrame:
        matrix = pd.DataFrame(0, index=index, columns=index)
        for i in index:
            matrix.loc[i : i + embargo_time, i] = 1
        return matrix

    @staticmethod
    def seq_bootstrap(ind_matrix: pd.DataFrame, sample_length: int) -> np.ndarray:
        phi = []
        while len(phi) < sample_length:
            avg_uniqueness = ind_matrix[phi].sum(axis=1) if phi else ind_matrix.sum(axis=1)
            probs = avg_uniqueness / avg_uniqueness.sum()
            phi.append(np.random.choice(ind_matrix.columns, p=probs))
        return np.array(phi)


class FeatureSampling:
    """Feature sampling helpers."""

    @staticmethod
    def get_features_at_events(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        return features.loc[events.index]
