"""Fractional differencing utilities for stationarity.

Functions
---------
- ``getWeights(d, size)``
    Compute weights for fractional differencing.
- ``fracDiff(series, d, thres=.01)``
    Apply fractional differencing to a series.
- ``getWeights_FFD(d, thres)``
    Weights for the fixed width fractional differencing method.
- ``fracDiff_FFD(series, d, thres)``
    Fixed width fractional differencing.
- ``getOptimalFFD(series, tau, maxLag, **kargs)``
    Brute force search for the smallest ``d`` passing ADF test.
- ``mpGetOptimalFFD(series, taus, **kargs)``
    Multiprocessing wrapper for ``getOptimalFFD``.
- ``OptimalFFD(series, **kargs)``
    Convenience function returning optimal differencing order.
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
#      Fractionally Differentiated Features
# =================================================================================================================

def getWeights(d, size):
    """Calculate weights for fractional differencing.

    Parameters
    ----------
    d : float
        Differencing order.
    size : int
        Number of weights to compute. Larger values approximate longer memory.
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[:: -1]).reshape(-1, 1)
    return w

def fracDiff(series, d, thres = .01):
    """Apply fractional differencing while dropping insignificant weights.

    Parameters
    ----------
    series : pandas.Series
        Price sequence to difference.
    d : float
        Differencing order.
    thres : float
        Threshold for weight significance. Computation stops when the
        cumulative weight loss exceeds ``thres``.
    """
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])  # each obs has a weight
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))  # cumulative weights
    w_ /= w_[-1]  # determine the relative weight-loss
    skip = w_[w_ > thres].shape[0]  # the no. of results where the weight-loss is beyond the acceptable value
    # 3) Apply weights to values
    df = {}  # empty dictionary
    for name in series.columns:
        # fill the na prices
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()  # create a pd series
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]  # find the iloc th obs

            test_val = series.loc[loc, name]  # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()

            if not np.isfinite(test_val).any():
                continue  # exclude NAs
            try:  # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
                df_.loc[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
            except:
                continue
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def getWeights_FFD(d, thres):
    # thres>0 drops insignificant weights
    w = [1.]
    k = 1
    while abs(w[-1]) >= thres:
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    w = np.array(w[:: -1]).reshape(-1, 1)[1:]
    return w

def fracDiff_FFD(series, d, thres=1e-5):
    """
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    # w = getWeights(d, series.shape[0])
    # w=getWeights_FFD(d,thres)
    width = len(w) - 1
    # 2) Apply weights to values
    df = {}  # empty dict
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()  # empty pd.series
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            # try: # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
            df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0: loc1])[0, 0]
            # except:
            #     continue

        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def getOptimalFFD(data, start = 0, end = 1, interval = 10, t = 1e-5):
    """

    :param data:
    :param start:
    :param end:
    :param interval:
    :param t:
    :return:
    """
    d = np.linspace(start, end, interval)
    out = mpJobList(mpGetOptimalFFD, ('molecules', d), redux=pd.DataFrame.append, data=data)

    return out

def mpGetOptimalFFD(data, molecules, t=1e-5):
    cols = ['adfStat', 'pVal', 'lags', 'nObs', '95% conf']
    out = pd.DataFrame(columns=cols)

    for d in molecules:
        try:
            dfx = fracDiff_FFD(data.to_frame(), d, thres=t)
            dfx = sm.tsa.stattools.adfuller(dfx['price'], maxlag=1, regression='c', autolag=None)
            out.loc[d] = list(dfx[:4]) + [dfx[4]['5%']]
        except Exception as e:
            print(f'{d} error: {e}')
    return out

def OptimalFFD(data, start=0, end=1, interval=10, t=1e-5):
    for d in np.linspace(start, end, interval):
        dfx = fracDiff_FFD(data.to_frame(), d, thres=t)
        if sm.tsa.stattools.adfuller(dfx['price'], maxlag=1, regression='c', autolag=None)[1] < 0.05:
            return d
    print('no optimal d')
    return d


__all__ = ['getWeights','fracDiff','getWeights_FFD','fracDiff_FFD','getOptimalFFD','mpGetOptimalFFD','OptimalFFD']
