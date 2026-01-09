"""Lightweight multiprocessing helpers."""

from __future__ import annotations

from typing import Callable, Iterable, List

import pandas as pd
from joblib import Parallel, delayed


def mp_pandas_obj(
    func: Callable,
    pd_obj: pd.DataFrame | pd.Series,
    axis: int = 0,
    n_jobs: int = -1,
) -> pd.Series:
    """Apply a function to pandas object rows/columns in parallel."""
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (columns) or 1 (rows)")

    if axis == 1:
        iterator = (pd_obj.iloc[i] for i in range(len(pd_obj)))
    else:
        iterator = (pd_obj.iloc[:, i] for i in range(pd_obj.shape[1]))

    results = Parallel(n_jobs=n_jobs)(delayed(func)(chunk) for chunk in iterator)
    return pd.Series(results)
