"""Sampling utilities for constructing various bar types.

Functions
---------
- ``BarSampling(df, column, threshold, tick=False)``
    Sample bars based on price, volume or dollar value thresholds.
- ``get_ratio(df, column, n_ticks)``
    Compute the ratio of cumulative value to tick count.
- ``select_sample_data(ref, sub, price_col, date)``
    Extract tick and subordinate data for a specific date.
- ``count_bars(df, price_col='price')``
    Count weekly sampled bars.
- ``scale(s)``
    Normalize a series between 0 and 1.
- ``getReturns(s)``
    Log return series of ``s``.
- ``get_test_stats(bar_types, bar_returns, test_func, *args, **kwds)``
    Apply statistical tests to bar returns.
- ``df_rolling_autocorr(df, window, lag=1)``
    Rolling autocorrelation of a DataFrame.
- ``signed_tick(tick)``
    Signed tick indicator used for imbalance bars.
- ``tick_imbalance_bar(tick, ...)``
    Create bars when cumulative signed tick exceeds expectation.
- ``tick_runs_bar(tick, ...)``
    Construct run bars based on tick imbalance.
- ``volume_runs_bar(tick, ...)``
    Run bars using volume information.
- ``getRunBars(tick, ...)``
    Generic run bar generator for any ticker column.
- ``getSequence(p0, p1, bs)``
    Sequence sign used in imbalance calculations.
- ``getImbalance(t)``
    Return cumulative imbalance sign.
- ``test_t_abs(absTheta, t, E_bs)``
    Check if imbalance exceeds expectation.
- ``getAggImalanceBar(df)``
    Aggregate imbalance bars from a DataFrame.
- ``getRolledSeries(series, dictio)``
    Apply roll gaps to futures series.
- ``rollGaps(series, dictio, matchEnd=True)``
    Compute gaps between contract rolls.
- ``getBollingerRange(data, window=21, width=0.005)``
    Compute Bollinger band parameters.
- ``pcaWeights(cov, riskDist=None, risktarget=1.0, valid=False)``
    Risk allocation weights following PCA.
- ``CusumEvents(df, limit)``
    Event sampling based on a CUSUM filter.
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
def BarSampling(df, column, threshold, tick = False):
    """Sample bars from time series data.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from ``getDataFrame``.
    column : str
        Column used as sampling criterion: ``'price'`` (time bar), ``'v'``
        (volume) or ``'dv'`` (dollar value).
    tick : bool, optional
        If ``True`` sample by tick count instead of the column value.

    Hyperparameters
    ---------------
    threshold : float
        Sample once the cumulative value exceeds this amount.

    Returns
    -------
    pandas.DataFrame
        Sampled bar data.
    """
    t = df[column]
    ts = 0
    idx = []
    if tick:
        for i, x in enumerate(t):
            ts += 1
            if ts >= threshold:
                idx.append(i)
                ts = 0
    else:
        for i, x in enumerate(t):
            ts += x
            if ts >= threshold:
                idx.append(i)
                ts = 0
    return df.iloc[idx].drop_duplicates()


def get_ratio(df, column, n_ticks):
    """Return the ratio of cumulative value to number of ticks.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from ``BarSampling``.
    column : str
        Field used for the ratio, e.g. ``'dollar_value'`` or ``'volume'``.
    n_ticks : int
        Number of ticks in the sample.
    """
    return df[column].sum() / n_ticks


def select_sample_data(ref, sub, price_col, date):
    """Select data for a specific date from reference and sub datasets.

    Parameters
    ----------
    ref : pandas.DataFrame
        DataFrame containing tick data.
    sub : pandas.DataFrame
        Subordinated price series.
    price_col : str
        Name of the price column.
    date : str
        Date to retrieve in ``YYYY-MM-DD`` format.

    Returns
    -------
    tuple(pd.Series, pd.Series)
        Series from ``ref`` and ``sub`` for the given date.
    """

    xdf = ref[price_col].loc[date]
    xtdf = sub[price_col].loc[date]

    return xdf, xtdf

def count_bars(df, price_col = 'price'):
    """Count how many bars are sampled each week."""
    return df.groupby(pd.Grouper(freq='1W'))[price_col].count()

def scale(s):
    """Scale a series to the 0-1 range for comparison."""
    return (s - s.min()) / (s.max() - s.min())

def getReturns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))

def get_test_stats(bar_types, bar_returns, test_func, *args, **kwds):
    dct = {bar: (int(bar_ret.shape[0]), test_func(bar_ret, *args, **kwds))
           for bar, bar_ret in zip(bar_types, bar_returns)}
    df = (pd.DataFrame.from_dict(dct)
          .rename(index={0: 'sample size', 1: f'{test_func.__name__}_stat'}).T)
    return df

def df_rolling_autocorr(df, window, lag = 1):
    """Compute rolling column-wise autocorrelation for a DataFrame."""
    return (df.rolling(window = window).corr(df.shift(lag)))

def signed_tick(tick, initial_value=1.0):
    diff = tick['price'] - tick['price'].shift(1)
    return (abs(diff) / diff).ffill().fillna(initial_value)

def tick_imbalance_bar(tick, initial_expected_bar_size = 150, initial_expected_signed_tick = .1,
                       lambda_bar_size = .1, lambda_signed_tick = .1):
    tick = tick.sort_index(ascending = True)
    tick = tick.reset_index()

    # Part 1. Determine bar numbers from tick imbalance
    tick_imbalance = signed_tick(tick).cumsum().values
    tick_imbalance_group = []

    expected_bar_size = initial_expected_bar_size
    expected_signed_tick = initial_expected_signed_tick
    expected_tick_imbalance = expected_bar_size * expected_signed_tick

    current_group = 1
    previous_i = 0

    for i in range(len(tick)):
        tick_imbalance_group.append(current_group)

        if abs(tick_imbalance[i]) >= abs(expected_tick_imbalance):  # EMA-style update
            expected_bar_size = (lambda_bar_size * (i - previous_i + 1) + (1 - lambda_bar_size) * expected_bar_size)
            expected_signed_tick = (lambda_signed_tick * tick_imbalance[i] /
                                    (i - previous_i + 1) + (1 - lambda_signed_tick) * expected_signed_tick)
            expected_tick_imbalance = expected_bar_size * expected_signed_tick

            tick_imbalance -= tick_imbalance[i]

            previous_i = i
            current_group += 1

    # Part 2. Create OHLCV bars based on bar numbering
    tick['tick_imbalance_group'] = tick_imbalance_group
    groupby = tick.groupby('tick_imbalance_group')

    bars = groupby['price'].ohlc()
    bars[['volume', 'value']] = groupby[['volume', 'value']].sum()
    bars['t'] = groupby['t'].first()

    bars.set_index('t', inplace=True)

    return bars

def tick_runs_bar(tick, initial_expected_bar_size, initial_buy_prob,
                  lambda_bar_size=.1, lambda_buy_prob=.1):
    tick = tick.sort_index(ascending=True)
    tick = tick.reset_index()
    _signed_tick = signed_tick(tick)
    imbalance_tick_buy = _signed_tick.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_tick_sell = _signed_tick.apply(lambda v: -v if v < 0 else 0).cumsum()
    group = []
    expected_bar_size = initial_expected_bar_size
    buy_prob = initial_buy_prob
    expected_runs = expected_bar_size * max(buy_prob, 1 - buy_prob)
    current_group = 1
    previous_i = 0
    for i in range(len(tick)):
        group.append(current_group)

        if max(imbalance_tick_buy[i], imbalance_tick_sell[i]) >= expected_runs:
            expected_bar_size = (lambda_bar_size * (i - previous_i + 1) + (1 - lambda_bar_size) * expected_bar_size)
            buy_prob = (lambda_buy_prob * imbalance_tick_buy[i] /
                        (i - previous_i + 1) + (1 - lambda_buy_prob) * buy_prob)
            previous_i = i
            imbalance_tick_buy -= imbalance_tick_buy[i]
            imbalance_tick_sell -= imbalance_tick_sell[i]
            current_group += 1
    tick['group'] = group
    groupby = tick.groupby('group')
    bars = groupby['price'].ohlc()
    bars[['volume', 'value']] = groupby[['volume', 'value']].sum()
    bars['t'] = groupby['t'].first()
    bars.set_index('t', inplace=True)
    return bars

def volume_runs_bar(tick, initial_expected_bar_size, initial_buy_prob, initial_buy_volume,
                    initial_sell_volume, lambda_bar_size=.1, lambda_buy_prob=.1,
                    lambda_buy_volume=.1, lambda_sell_volume=.1):
    tick = tick.sort_index(ascending=True)
    tick = tick.reset_index()
    _signed_tick = signed_tick(tick)
    _signed_volume = _signed_tick * tick['volume']
    imbalance_tick_buy = _signed_tick.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_volume_buy = _signed_volume.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_volume_sell = _signed_volume.apply(lambda v: v if -v < 0 else 0).cumsum()

    group = []

    expected_bar_size = initial_expected_bar_size
    buy_prob = initial_buy_prob
    buy_volume = initial_buy_volume
    sell_volume = initial_sell_volume
    expected_runs = expected_bar_size * max(buy_prob * buy_volume, (1 - buy_prob) * sell_volume)

    current_group = 1
    previous_i = 0
    for i in range(len(tick)):
        group.append(current_group)

        if max(imbalance_volume_buy[i], imbalance_volume_sell[i]) >= expected_runs:
            expected_bar_size = (lambda_bar_size * (i - previous_i + 1) + (1 - lambda_bar_size) * expected_bar_size)
            buy_prob = (lambda_buy_prob * imbalance_tick_buy[i] /
                        (i - previous_i + 1) + (1 - lambda_buy_prob) * buy_prob)
            buy_volume = (lambda_buy_volume * imbalance_volume_buy[i] + (1 - lambda_buy_volume) * buy_volume)
            sell_volume = (lambda_sell_volume * imbalance_volume_sell[i] + (1 - lambda_sell_volume) * sell_volume)
            previous_i = i
            imbalance_tick_buy -= imbalance_tick_buy[i]
            imbalance_volume_buy -= imbalance_volume_buy[i]
            imbalance_volume_sell -= imbalance_volume_sell[i]
            current_group += 1
    tick['group'] = group
    groupby = tick.groupby('group')
    bars = groupby['price'].ohlc()
    bars[['volume', 'value']] = groupby[['volume', 'value']].sum()
    bars['t'] = groupby['t'].first()
    bars.set_index('t', inplace=True)
    return bars

class FuturesRollETFTrick:
    def __init__(self, df):
        self.data_index = df.index
        df['pinpoint'] = ~(df['symbol'] == df['symbol'].shift(1))
        df = df.reset_index()
        df['Diff_close'] = df['close'].diff()
        df['Diff_close_open'] = df['close'] - df['open']
        df['H_part'] = 1/df['open'].shift(-1)
        self.prev_h = 1
        self.prev_k = 1
        _ = zip(df.H_part, df.Diff_close, df.Diff_close_open, df.pinpoint)
        df['K'] = [self.process_row(x, y, z, w) for x, y, z, w in _]
        self.data = df['K'].values

    @property
    def series(self):
        return pd.Series(self.data, index=self.data_index)

    def process_row(self, h_part, diff_close, diff_open_close, pinpoint):
        if pinpoint:
            h = self.prev_k*h_part
            delta = diff_open_close
        else:
            h = self.prev_h
            delta = diff_close
        k = self.prev_k + h*delta
        self.prev_h = h
        self.prev_k = k
        return k

def getRunBars(tick, initial_expected_bar_size, initial_buy_prob, initial_buy_volume, initial_sell_volume,
               ticker = 'volume', lambda_bar_size=.1, lambda_buy_prob=.1, lambda_buy_volume=.1, lambda_sell_volume=.1):
    tick = tick.sort_index(ascending=True)
    tick = tick.reset_index()
    _signed_tick = signed_tick(tick)
    _signed_volume = _signed_tick * tick[ticker]
    imbalance_tick_buy = _signed_tick.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_volume_buy = _signed_volume.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_volume_sell = _signed_volume.apply(lambda v: v if -v < 0 else 0).cumsum()

    group = []

    expected_bar_size = initial_expected_bar_size
    buy_prob = initial_buy_prob
    buy_volume = initial_buy_volume
    sell_volume = initial_sell_volume
    expected_runs = expected_bar_size * max(buy_prob * buy_volume, (1 - buy_prob) * sell_volume)

    current_group = 1
    previous_i = 0
    for i in range(len(tick)):
        group.append(current_group)

        if max(imbalance_volume_buy[i], imbalance_volume_sell[i]) >= expected_runs:
            expected_bar_size = (lambda_bar_size * (i - previous_i + 1) + (1 - lambda_bar_size) * expected_bar_size)
            buy_prob = (lambda_buy_prob * imbalance_tick_buy[i] /
                        (i - previous_i + 1) + (1 - lambda_buy_prob) * buy_prob)
            buy_volume = (lambda_buy_volume * imbalance_volume_buy[i] + (1 - lambda_buy_volume) * buy_volume)
            sell_volume = (lambda_sell_volume * imbalance_volume_sell[i] + (1 - lambda_sell_volume) * sell_volume)
            previous_i = i
            imbalance_tick_buy -= imbalance_tick_buy[i]
            imbalance_volume_buy -= imbalance_volume_buy[i]
            imbalance_volume_sell -= imbalance_volume_sell[i]
            current_group += 1
    tick['group'] = group
    groupby = tick.groupby('group')
    bars = groupby['price'].ohlc()
    bars[ticker] = groupby[ticker].sum()
    bars['value'] = groupby['value'].sum()
    #bars[['volume', 'value']] = groupby[['volume', 'value']].sum()
    bars['t'] = groupby['t'].first()
    bars.set_index('t', inplace=True)
    return bars

@jit(nopython=True)
def getSequence(p0,p1,bs):
    if numba_isclose((p1-p0),0.0,abs_tol=0.001):
        return bs[-1]
    else: return np.abs(p1-p0)/(p1-p0)

@jit(nopython=True)
def getImbalance(t):
    """Noted that this function return a list start from the 2nd obs"""
    bs = np.zeros_like(t)
    for i in np.arange(1,bs.shape[0]):
        bs[i-1] = getSequence(t[i-1],t[i],bs[:i-1])
    return bs[:-1] # remove the last value

def test_t_abs(absTheta, t, E_bs):
    """
    Bool function to test inequality
    * row is assumed to come from df.itertuples()
    - absTheta: float(), row.absTheta
    - t: pd.Timestamp
    - E_bs: float, row.E_bs
    """
    return (absTheta >= t * E_bs)

def getAggImalanceBar(df):
    """
    Implements the accumulation logic
    """
    start = df.index[0]
    bars = []
    for row in df.itertuples():
        t_abs = row.absTheta
        rowIdx = row.Index
        E_bs = row.E_bs

        t = df.loc[start:rowIdx].shape[0]
        if t < 1: t = 1  # if t less than 1, set equal to 1
        if test_t_abs(t_abs, t, E_bs):
            bars.append((start, rowIdx, t))
            start = rowIdx
    return bars

def getRolledSeries(series, dictio):
    gaps = rollGaps(series, dictio)
    for field in ['Close', 'Volume']:
        series[field] -= gaps
    return series

def rollGaps(series, dictio, matchEnd=True):
    # Compute gaps at each roll, between previous close and next open
    rollDates = series[dictio['Instrument']].drop_duplicates(keep='first').index
    gaps = series[dictio['Close']] * 0
    iloc = list(series.index)
    iloc = [iloc.index(i) - 1 for i in rollDates] # index of days prior to roll
    gaps.loc[rollDates[1:]] = series[dictio['Open']].loc[rollDates[1:]] - series[dictio['Close']].iloc[iloc[1:]].values
    gaps = gaps.cumsum()
    if matchEnd:
        gaps -= gaps.iloc[-1]
    return gaps

def getBollingerRange(data: pd.Series, window: int = 21, width: float = 0.005):
    """Return parameters required to build Bollinger Bands.

    Parameters
    ----------
    data : pandas.Series
        Price series.
    window : int, optional
        Rolling window span.
    width : float, optional
        Width multiplier for the band.
    """
    avg = data.ewm(span = window).mean()
    std0 = avg * width
    lower = avg - std0
    upper = avg + std0

    return avg, upper, lower, std0

def pcaWeights(cov, riskDist = None, risktarget = 1.0, valid = False):
    """Match risk targets according to a risk allocation distribution.

    Parameters
    ----------
    cov : pandas.DataFrame
        Covariance matrix.
    riskDist : array-like, optional
        Custom risk distribution. ``None`` allocates all risk to the smallest
        eigenvalue component.
    risktarget : float, optional
        Scaling factor for ``riskDist``.
    valid : bool, optional
        If ``True`` also return the contribution of each component.
    """
    eVal, eVec = np.linalg.eigh(cov)  # Hermitian Matrix
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1.

    loads = riskTarget * (riskDist / eVal) ** 0.5
    wghts = np.dot(eVec, np.reshape(loads, (-1, 1)))

    if vaild == True:
        ctr = (loads / riskTarget) ** 2 * eVal  # verify riskDist contribution
        return (wghts, ctr)
    else:
        return wghts

def CusumEvents(df: pd.Series, limit: float):
    """Event based sampling using a CUSUM filter.

    Parameters
    ----------
    df : pandas.Series
        Price series.
    limit : float
        Threshold for the CUSUM filter. Higher values yield fewer events.
    """
    idx, _up, _dn = [], 0, 0
    diff = df.diff()
    for i in range(len(diff)):
        if _up + diff.iloc[i] > 0:
            _up = _up + diff.iloc[i]
        else:
            _up = 0

        if _dn + diff.iloc[i] < 0:
            _dn = _dn + diff.iloc[i]
        else:
            _dn = 0

        if _up > limit:
            _up = 0;
            idx.append(i)
        elif _dn < - limit:
            _dn = 0;
            idx.append(i)
    return idx

__all__ = ['BarSampling','get_ratio','select_sample_data','count_bars','scale','getReturns','get_test_stats','df_rolling_autocorr','signed_tick','tick_imbalance_bar','tick_runs_bar','volume_runs_bar','getRunBars','getSequence','getImbalance','test_t_abs','getAggImalanceBar','getRolledSeries','rollGaps']
