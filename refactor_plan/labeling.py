"""Label creation and sample-weight utilities used in the notebooks.

Functions
---------
- ``getDailyVolatility(close, span=100)``
    Estimate exponentially weighted daily volatility.
- ``addVerticalBarrier(tEvents, close, numDays=1)``
    Place a vertical barrier ``numDays`` after each event.
- ``tradableHour(i, start='09:40', end='15:50')``
    Check whether timestamp ``i`` is within trading hours.
- ``getTEvents(gRaw, h, symmetric=True, isReturn=False)``
    Symmetric or asymmetric CUSUM filter.
- ``getTripleBarrier(close, events, ptSl, molecule)``
    Apply the triple barrier labeling method.
- ``getEvents(...)``
    Form a DataFrame of events for labeling.
- ``getBins(events, close)``
    Compute event outcome labels.
- ``dropLabels(events, minPct=0.05)``
    Drop labels that are too rare.
- ``getSampleWeights`` and ``mpSampleWeights``
    Calculate sample weights based on concurrent events.
- ``getIndMatrix`` and ``getAvgUniqueness``
    Utilities for sequential bootstrapping.
- ``seqBootstrap(indM, sLength=None)``
    Draw sample indices with sequential uniqueness weighting.
- ``getBollingerBand``
    Simple Bollinger Band helper.
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
def getDailyVolatility(close, span = 100):
    """Estimate daily rolling volatility.

    Parameters
    ----------
    close : pandas.Series
        Price series.
    span : int, optional
        Number of days for the exponential moving window (default ``100``).
    """
    # daily vol reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1],
                     index=close.index[close.shape[0] - df0.shape[0]:]))
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0 = df0.ewm(span = span).std().rename('dailyVol')
    return df0

def addVerticalBarrier(tEvents, close, numDays=1):
    """Create a vertical barrier after a fixed number of days.

    Parameters
    ----------
    tEvents : pd.DatetimeIndex
        Event timestamps from ``getTEvents``.
    close : pandas.Series
        Price series.
    numDays : int, optional
        Number of days after which the barrier is placed (default ``1``).
    """
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1


def tradableHour(i, start='09:40', end='15:50'):
    """
    : param i: a datetimeIndex value
    : param start: the start of the trading hour
    : param end: the end of the trading hour

    : return: bool, is tradable hour or not"""
    time = i.strftime('%H:%M')
    return (time < end and time > start)

def getTEvents(gRaw, h, symmetric=True, isReturn=False):
    """
    Symmetric CUSUM Filter
    Sample a bar t iff S_t >= h at which point S_t is reset
    Multiple events are not triggered by gRaw hovering around a threshold level
    It will require a full run of length h for gRaw to trigger an event

    Two arguments:
        gRaw: the raw time series we wish to filter (gRaw), e.g. return
        h: threshold

    Return:
        pd.DatatimeIndex.append(tEvents):
    """
    tEvents = []
    if isReturn:
        diff = gRaw
    else:
        diff = gRaw.diff()
    if symmetric:
        sPos, sNeg = 0, 0
        if np.shape(h) == ():

            for i in diff.index[1:]:
                sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
                if sNeg < -h and tradableHour(i):
                    sNeg = 0;
                    tEvents.append(i)
                elif sPos > h and tradableHour(i):
                    sPos = 0;
                    tEvents.append(i)
        else:
            for i in diff.index[1:]:
                sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
                if sNeg < -h[i] and tradableHour(i):
                    sNeg = 0;
                    tEvents.append(i)
                elif sPos > h[i] and tradableHour(i):
                    sPos = 0;
                    tEvents.append(i)
    else:
        sAbs = 0
        if np.shape(h) == ():

            for i in diff.index[1:]:
                sAbs = sAbs + diff.loc[i]
                if sAbs > h and tradableHour(i):
                    sAbs = 0;
                    tEvents.append(i)

        else:
            for i in diff.index[1:]:
                sAbs = sAbs + diff.loc[i]
                if sAbs > h[i] and tradableHour(i):
                    sAbs = 0;
                    tEvents.append(i)

    return pd.DatetimeIndex(tEvents)

def getTripleBarrier(close, events, ptSl, molecule):
    """Implement the triple barrier method.

    Labels are assigned when either the horizontal or vertical barrier is
    touched.

    Parameters
    ----------
    close : pandas.Series
        Price series.
    events : pandas.DataFrame
        Must contain columns ``t1`` (vertical barrier timestamps) and ``trgt``
        (unit width of the horizontal barrier).
    ptSl : list-like of two floats
        Multipliers for the upper and lower barriers. ``0`` disables the barrier.
    molecule : list-like
        Subset of event indices processed by a single thread.
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs

    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc: t1]  # price path
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # return path
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking

    return out


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    """Form events for the triple barrier method.

    Parameters
    ----------
    close : pandas.Series
        Price series.
    tEvents : pd.DatetimeIndex
        Seed timestamps for barriers.
    ptSl : list-like
        Multipliers for profit-taking and stop-loss barriers.
    trgt : pandas.Series
        Target returns.
    minRet : float
        Minimum target return required.
    numThreads : int
        Number of threads used by ``mpPandasObj``.
    t1 : pandas.Series, optional
        Predefined vertical barrier timestamps.
    side : pandas.Series, optional
        Side information for meta labeling.
    """
    # 1) determine the target returns
    for i in tEvents:
        if i not in trgt.index:
            trgt[str(i)] = np.NaN
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet

    # 2) compute vertical barrier (maximum holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)

    # 3) apply stop loss to form the event object
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[side.index & trgt.index], ptSl[:2]

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis = 1).dropna(subset = ['trgt'])
    df0 = mpPandasObj(func = getTripleBarrier, pdObj=('molecule', events.index),
                          numThreads = numThreads, close = close, events = events, ptSl = np.array(ptSl_))
    events['t1'] = df0.dropna(how = 'all').min(axis = 1)  # pd.min ignores nan

    if side is None:
        events = events.drop('side', axis=1)

    return events

def getBins(events, close):
    """Compute event outcome bins, including meta-labels when available.

    Parameters
    ----------
    events : pandas.DataFrame
        Must contain columns ``t1`` (end time), ``trgt`` (target) and
        optionally ``side`` (position direction).
    close : pandas.Series
        Price series.

    Returns
    -------
    pandas.DataFrame
        Contains the event returns and bin labels.
    """

    # 1) align prices with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    # 2) create output object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    if 'side' in events_: out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling

    return out


def dropLabels(events, minPct=0.05):
    # Drop labels with few observations by applying class weighting
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print('dropped label', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events



def getBinsNew(events, close, t1=None):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    -t1 is original vertical barrier series
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_: out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])

    if 'side' not in events_:
        # only applies when not meta-labeling
        # to update bin to 0 when vertical barrier is touched, we need the original
        # vertical barrier series since the events['t1'] is the time of first 
        # touch of any barrier and not the vertical barrier specifically. 
        # The index of the intersection of the vertical barrier values and the 
        # events['t1'] values indicate which bin labels needs to be turned to 0
        vtouch_first_idx = events[events['t1'].isin(t1.values)].index
        out.loc[vtouch_first_idx, 'bin'] = 0.

    if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out

def get_up_cross(df):
    """Return indices where the fast series crosses above the slow series."""
    crit1 = df.fast.shift(1) < df.slow.shift(1)
    crit2 = df.fast > df.slow
    return df.fast[(crit1) & (crit2)]

def get_down_cross(df):
    """Return indices where the fast series crosses below the slow series."""
    crit1 = df.fast.shift(1) > df.slow.shift(1)
    crit2 = df.fast < df.slow
    return df.fast[(crit1) & (crit2)]

def getUpCross(df, col):
    # col is price column
    crit1 = df[col].shift(1) < df.upper.shift(1)
    crit2 = df[col] > df.upper
    return df[col][(crit1) & (crit2)]

def getDownCross(df, col):
    # col is price column
    crit1 = df[col].shift(1) > df.lower.shift(1)
    crit2 = df[col] < df.lower
    return df[col][(crit1) & (crit2)]

# ===============================================================================================================
#           Sample Weights
# =================================================================================================================

def getConcurrentBar(closeIdx, t1, molecule):
    """Count concurrent events for each bar to compute uniqueness.

    Parameters
    ----------
    closeIdx : pd.Index
        Index of price bars.
    t1 : pd.Series
        Series of event end times (vertical barriers).
    molecule : list-like
        Event timestamps for which to compute the counts. Events occurring
        before ``molecule[0]`` or after ``molecule[-1]`` are ignored.

    Returns
    -------
    pandas.Series
        Number of concurrent events for each bar in ``molecule``.
    """
    # 1) search for events between molecule[0] and molecule[-1]
    # fill the unclosed events with the last available (index) date
    t1 = t1.fillna(closeIdx[-1])  # uncovered events impact other weights
    t1 = t1[t1 >= molecule[0]]  # events starting after the first molecule date
    # events that end before or at t1[molecule].max()
    t1 = t1.loc[: t1[molecule].max()]

    # 2) count how many events overlap each bar
    # find the indices begining start date ([t1.index[0]) and the furthest stop date (t1.max())
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    # form a 0-array, index: from the begining start date to the furthest stop date
    count = pd.Series(0, index=closeIdx[iloc[0]: iloc[1] + 1])
    # for each signal t1 (index: eventStart, value: eventEnd)
    for tIn, tOut in t1.iteritems():
        # add 1 if and only if [t_(i,0), t_(i.1)] overlaps with [t-1,t]
        count.loc[tIn: tOut] += 1  # every timestamp between tIn and tOut
    # compute the number of labels concurrents at t
    return count.loc[molecule[0]: t1[molecule].max()]  # only return the timespan of the molecule


def getAvgLabelUniq(t1, numCoEvents, molecule):
    """
    :param t1: pd series, timestamps of the vertical barriers. (index: eventStart, value: eventEnd).
    :param numCoEvent: 
    :param molecule: the date of the event on which the weight will be computed
        + molecule[0] is the date of the first event on which the weight will be computed
        + molecule[-1] is the date of the last event on which the weight will be computed
    :return
        wght: pd.Series, the sample weight of each (volume) bar
    """
    # derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    # for each events
    for tIn, tOut in t1.loc[wght.index].iteritems():
        # tIn, starts of the events, tOut, ends of the events
        # the more the coEvents, the lower the weights
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn: tOut]).mean()
    return wght


def mpSampleWeights(close, events, numThreads):
    """
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param numThreads: constant, The no. of threads concurrently used by the function
    :return
        wght: pd.Series, the sample weight of each (volume) bar
    """
    out = events[['t1']].copy(deep=True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index,
                                  t1=out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(getAvgLabelUniq, ('molecule', events.index), numThreads, t1=out['t1'],
                                numCoEvents = numCoEvents)
    return out


def getBollingerBand(price, window = None, width = None, numsd = None):
    """Construct a Bollinger Band for a price series.

    Parameters
    ----------
    price : pandas.Series
        Price series.
    window : int, optional
        Rolling window size.
    width : float, optional
        Upper/lower band width expressed as a ratio.
    numsd : float, optional
        Upper/lower band width expressed in standard deviations.
    """
    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1 + width)
        dnband = ave * (1 - width)
        return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)
    if numsd:
        upband = ave + (sd * numsd)
        dnband = ave - (sd * numsd)
        return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)


def getSampleWeights(close, events, numThreads):
    """
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param numThreads: constant, The no. of threads concurrently used by the function
    :return wght: pd.Series, the sample weight of each (volume) bar
    """
    out = events[['t1']].copy(deep=True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index, t1=out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(mpSampleWeights, ('molecule', events.index), numThreads, t1=out['t1'], numCoEvents=numCoEvents)
    return out


def getIndMatrix(barIx, t1):
    """Construct an indicator matrix linking observations to price bars.

    Parameters
    ----------
    barIx : sequence
        Indices of bars.
    t1 : pandas.Series
        Vertical barrier timestamps (index: event start, value: event end).

    Returns
    -------
    pandas.DataFrame
        Binary matrix showing which bars affect each observation.
    """
    indM = pd.DataFrame(0, index = barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()):  # signal = obs
        indM.loc[t0: t1, i] = 1.  # each obs each column, you can see how many bars are related to an obs/
    return indM


def getAvgUniqueness(indM):
    """Return the average uniqueness for each observation.

    Parameters
    ----------
    indM : pandas.DataFrame
        Indicator matrix created by ``getIndMatrix``.

    Returns
    -------
    float
        Average uniqueness across observations.
    """
    # average uniqueness from the indicator matrix
    c = indM.sum(axis=1)  # concurrency, how many obs share the same bar
    u = indM.div(c, axis=0)  # uniqueness, the more obs share the same bar, the less important the bar is
    avgU = u[u > 0].mean()  # average uniquenessn
    return avgU


def seqBootstrap(indM, sLength=None):
    """
    Give the index of the features sampled by the sequential bootstrap
    :param indM: binary matrix, indicate what (price) bars influence the label for each observation
    :param sLength: optional, sample length, default: as many draws as rows in indM
    """
    # Generate a sample via sequential bootstrap
    if sLength is None:  # default
        sLength = indM.shape[1]  # sample length = # of rows in indM
    # Create an empty list to store the sequence of the draws
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()  # store the average uniqueness of the draw
        for i in indM:  # for every obs
            indM_ = indM[phi + [i]]  # add the obs to the existing bootstrapped sample
            # get the average uniqueness of the draw after adding to the new phi
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[
                -1]  # only the last is the obs concerned, others are not important
        prob = avgU / avgU.sum()  # cal prob <- normalise the average uniqueness
        phi += [np.random.choice(indM.columns, p=prob)]  # add a random sample from indM.columns with prob. = prob
    return phi


def main():
    # t0: t1.index; t1: t1.values
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
    # index of bars
    barIx = range(t1.max() + 1)
    # get indicator matrix
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    print(phi)
    print('Standard uniqueness:', getAvgUniqueness(indM[phi]).mean())
    phi = seqBootstrap(indM)
    print(phi)
    print('Sequential uniqueness:', getAvgUniqueness(indM[phi]).mean())

def getRndT1(numObs, numBars, maxH):
    # random t1 Series
    t1 = pd.Series()
    for _ in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()


def auxMC(numObs, numBars, maxH):
    # Parallelized auxiliary function
    t1 = getRndT1(numObs, numBars, maxH)
    barIx = range(t1.max() + 1)
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return {'stdU': stdU, 'seqU': seqU}


def mainMC(numObs = 10, numBars = 100, maxH = 5, numIters = 1E6, numThreads = 24):
    # Monte Carlo experiments
    jobs = []
    for _ in range(int(numIters)):
        job = {'func': auxMC, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads = numThreads)
    print(pd.DataFrame(out).describe())
    return


def mpSampleW(t1, numCoEvents, close, molecule):
    # Derive sample weight by return attribution
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn: tOut] / numCoEvents.loc[tIn: tOut]).sum()
    return wght.abs()


def SampleW(close, events, numThreads):
    """
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param numThreads: constant, The no. of threads concurrently used by the function
    """
    out = events[['t1']].copy(deep=True)
    numCoEvents = mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index,
                              t1=events['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents=numCoEvents,
                           close=close)
    out['w'] *= out.shape[0] / out['w'].sum()  # normalised, sum up to sample size

    return out

def getConcurUniqueness(close, events, numThreads):
    out = events[['t1']].copy(deep = True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx = close.index, t1 = out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(getAvgLabelUniq, ('molecule', events.index), numThreads, t1=out['t1'], numCoEvents = numCoEvents)
    out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents = numCoEvents,
                           close=close)
    out['w'] *= out.shape[0] / out['w'].sum()  # normalised, sum up to sample size
    return out

def getTimeDecay(tW, clfLastW = 1.):
    """
    apply piecewise-linear decay to observed uniqueness (tW)
    clfLastW = 1: no time decay
    0 <= clfLastW <= 1: weights decay linearly over time, but every obersevation still receives a strictly positive weight
    c = 0: weughts converge linearly to 0 as they become older
    c < 0: the oldest portion cT of the observations receive 0 weight
    c > 1: weights increase as they get older"""
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()  # cumulative sum of the observed uniqueness
    if clfLastW >= 0:  # if 0 <= clfLastW <= 1
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:  # if -1 <= clfLastW < 1
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW

__all__ = ['getDailyVolatility','addVerticalBarrier','tradableHour','getTEvents','getTripleBarrier','getEvents','getBins','dropLabels','getBinsNew','get_up_cross','get_down_cross','getUpCross','getDownCross','getConcurrentBar','getAvgLabelUniq','mpSampleWeights','getBollingerBand','getSampleWeights','getIndMatrix','getAvgUniqueness','seqBootstrap','getRndT1','auxMC','mainMC','mpSampleW','SampleW','getConcurUniqueness','getTimeDecay']
