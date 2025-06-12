"""Plotting helpers for bar data and statistics.

Functions
---------
- ``plot_bar_counts(tick, volume, dollar)``
    Visualize bar counts over time.
- ``plot_hist(bar_types, bar_returns)``
    Histogram of returns for each bar type.
- ``plotSampleData(ref, sub, bar_type, *args, **kwds)``
    Overlay sampled data on reference prices.
- ``plot_autocorr(bar_types, bar_returns)``
    Autocorrelation plots for bar returns.
- ``plotWeights(dRange, nPlots, size)``
    Plot fractional differencing weights.
- ``plotMinFFD()``
    Search for minimum ``d`` passing the ADF test.
- ``plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs)``
    Plot feature importance results.
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
def plot_bar_counts(tick, volume, dollar):
    """Plot counts of tick, volume and dollar bars."""
    f, ax = plt.subplots(figsize=(15, 5))
    tick.plot(ax=ax, ls='-', label='tick count')
    volume.plot(ax=ax, ls='--', label='volume count')
    dollar.plot(ax=ax, ls='-.', label='dollar count')
    ax.set_title('Scaled Bar Counts')
    ax.legend()
    return

def plot_hist(bar_types, bar_returns):
    f, axes = plt.subplots(len(bar_types), figsize=(10, 6))
    for i, (bar, typ) in enumerate(zip(bar_returns, bar_types)):
        g = sns.distplot(bar, ax=axes[i], kde=False, label=typ)
        g.set(yscale='log')
        axes[i].legend()
    plt.tight_layout()
    return

def plotSampleData(ref, sub, bar_type, *args, **kwds):
    """Plot sampled bar data together with the original series."""

    f, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 7))
    ref.plot(*args, **kwds, ax=axes[0], label='price')
    sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
    axes[0].legend()
    ref.plot(*args, **kwds, ax=axes[1], marker='o', label='price')
    sub.plot(*args, **kwds, ax=axes[2], marker='X', ls='',
             color='r', label=bar_type)
    for ax in axes[1:]: ax.legend()
    plt.tight_layout()
    return

def plot_autocorr(bar_types, bar_returns):
    f, axes = plt.subplots(len(bar_types), figsize=(10, 7))

    for i, (bar, typ) in enumerate(zip(bar_returns, bar_types)):
        sm.graphics.tsa.plot_acf(bar, lags=120, ax=axes[i],
                                 alpha=0.05, unbiased=True, fft=True,
                                 zero=False,
                                 title=f'{typ} AutoCorr')
    plt.tight_layout()
    return

def plotWeights(dRange, nPlots, size):
    """Visualize fractional differencing weights over a range of ``d``."""
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[:: -1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left')
    plt.show()
    return

def plotMinFFD():
    """Search for the minimum ``d`` that passes the ADF test."""
    from statsmodels.tsa.stattools import adfuller
    path = './'
    instName = 'ES1_Index_Method12'
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    df0 = pd.read_csv(path + instName + '.csv', index_col=0, parse_dates=True)
    for d in np.linspace(0, 1, 11):
        df1 = np.log(df0[['Close']]).resample('1D').last()  # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, thres=.01)
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        df2 = adfuller(df2['Close'], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[: 4]) + [df2[4]['5%']] + [corr]  # with critical value
    out.to_csv(path + instName + '_testMinFFD.csv')
    out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.savefig(path + instName + '_testMinFFD.png')
    return

def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    plt.figure(figsize=(10, imp.shape[0] / 5.))
    # sort imp['mean'] from low to high
    imp = imp.sort_values('mean', ascending=True)
    # plot horizontal bar
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    if method == 'MDI':  # for MDI
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)  # disable y-axis
    for i, j in zip(ax.patches, imp.index):  # label
        ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center', va='center', color='black')
    plt.title(
        'tag=' + tag + ' | simNum= ' + str(simNum) + ' | oob=' + str(round(oob, 4)) + ' | oos=' + str(round(oos, 4)))
    plt.savefig(pathOut + 'featImportance_' + str(simNum) + '.png', dpi=100)
    plt.clf()
    plt.close()
    return

__all__ = ['plot_bar_counts','plot_hist','plotSampleData','plot_autocorr','plotWeights','plotMinFFD','plotFeatImportance','plotCorrMatrix']
