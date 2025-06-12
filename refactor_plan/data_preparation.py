"""Summary of functions in this module.

Functions
---------
- ``getDataFrame(df: pd.DataFrame) -> pd.DataFrame``
    Clean raw high frequency quotes into an OHLC style DataFrame. ``df`` should
    contain ``['price', 'buy', 'sell', 'volume']`` with a datetime index. The
    output adds ``v`` and ``dv`` columns where ``v`` is volume and ``dv`` is
    volume multiplied by price.

- ``numba_isclose(a: float, b: float, rel_tol=1e-9, abs_tol=0.0) -> bool``
    Compare two numeric values using relative and absolute tolerances. Returns
    ``True`` when ``|a-b|`` is within the tolerances. Example: ``0.1`` and
    ``0.1000001``.

- ``getOHLC(ref: pd.Series, sub: pd.Series) -> pd.DataFrame``
    Build OHLC bars from ``ref`` prices using the timestamps in ``sub``. The
    result contains columns ``['end','start','open','high','low','close']`` for
    each interval.

- ``madOutlier(y: Union[np.ndarray, pd.Series], thresh=3.0) -> np.ndarray``
    Detect outliers with the median absolute deviation method. ``y`` is a series
    of prices; the output is a boolean array flagging observations where the
    modified z-score exceeds ``thresh``.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import re
import os
import time
from collections import OrderedDict as od
import math
import sys
import datetime as dt
from datetime import timedelta
from random import gauss

import scipy.stats as stats
from scipy import interp
from scipy.stats import rv_continuous, kstest, norm
import scipy.cluster.hierarchy as sch

import copyreg, types, multiprocessing as mp
import copy
import platform
from multiprocessing import cpu_count

from numba import jit
from tqdm import tqdm, tqdm_notebook

import warnings
#statsmodels
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, log_loss, accuracy_score
from itertools import product
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import _BaseKFold
from sklearn import metrics
def getDataFrame(df):
    """Clean high-frequency quotes into an OHLC DataFrame.

    Returns
    -------
    pandas.DataFrame
        OHLC data including ``v`` (volume) and ``dv`` (dollar value).
    """
    temp = df[['price', 'buy', 'sell', 'volume']]
    temp['v'] = temp.volume
    temp['dv'] = temp.volume * temp.price
    temp.index = pd.to_datetime(temp.index)
    return temp

@jit(nopython = True)
def numba_isclose(a, b, rel_tol = 1e-09, abs_tol = 0.0):
    # rel_tol: relative tolerance
    # abs_tol: absolute tolerance
    return np.fabs(a-b) <= np.fmax(rel_tol*np.fmax(np.fabs(a),np.fabs(b)),abs_tol)

def getOHLC(ref, sub):
    """
    fn: get ohlc from custom

    # args
        ref: reference pandas series with all prices
        sub: custom tick pandas series

    # returns
        tick_df: dataframe with ohlc values
    """
    ohlc = []
    # for i in tqdm(range(sub.index.shape[0]-1)):
    for i in range(sub.index.shape[0] - 1):
        start, end = sub.index[i], sub.index[i + 1]
        tmp_ref = ref.loc[start:end]
        max_px, min_px = tmp_ref.max(), tmp_ref.min()
        o, h, l, c = sub.iloc[i], max_px, min_px, sub.iloc[i + 1]
        ohlc.append((end, start, o, h, l, c))
    cols = ['end', 'start', 'open', 'high', 'low', 'close']
    return (pd.DataFrame(ohlc, columns = cols))

@jit(nopython=True)
def madOutlier(y, thresh = 3.):
    """Detect outliers using the MAD method.

    Parameters
    ----------
    y : pandas.Series or np.ndarray
        Series of prices.
    thresh : float, optional
        Threshold for the modified z-score (default ``3.0``).

    Returns
    -------
    np.ndarray
        Boolean array indicating outlier positions.
    """
    median = np.median(y)
    print(median)
    diff = np.sum((y - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    print(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    print(modified_z_score)
    return modified_z_score > thresh

__all__ = ['getDataFrame', 'numba_isclose', 'getOHLC', 'madOutlier']
