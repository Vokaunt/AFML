"""Sample weighting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


class SampleWeights:
    """Weighting schemes for labeled events."""

    @staticmethod
    def get_time_decay(index: pd.Index, decay_factor: float = 0.5) -> np.ndarray:
        weights = np.linspace(decay_factor, 1.0, len(index))
        return weights / weights.sum()

    @staticmethod
    def get_concurrency_weights(t1: pd.Series, start_index: pd.Index) -> pd.Series:
        concurrency = pd.Series(0, index=start_index)
        for start, end in t1.items():
            concurrency.loc[start:end] += 1
        return 1 / concurrency.replace(0, np.nan)

    @staticmethod
    def compute_overlap_matrix(t1: pd.Series) -> pd.DataFrame:
        idx = t1.index
        matrix = pd.DataFrame(0, index=idx, columns=idx)
        for i, (start_i, end_i) in enumerate(t1.items()):
            for j, (start_j, end_j) in enumerate(t1.items()):
                if start_i <= end_j and start_j <= end_i:
                    matrix.iloc[i, j] = 1
        return matrix

    @staticmethod
    def compute_information_driven_weights(overlap_matrix: pd.DataFrame) -> pd.Series:
        concurrency = overlap_matrix.sum(axis=1)
        return 1 / concurrency.replace(0, np.nan)


class UniquenessSampling:
    """Uniqueness metrics for sampling."""

    @staticmethod
    def get_distance_based_uniqueness(features: pd.DataFrame) -> pd.Series:
        distances = pairwise_distances(features)
        uniqueness = distances.mean(axis=1)
        return pd.Series(uniqueness, index=features.index)
