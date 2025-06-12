"""Bet sizing utilities and supporting calculations.

Functions
---------
- ``avgActiveSignals_(signals, molecule)``
    Helper for multiprocessing average active signals.
- ``avgActiveSignals(signals)``
    Compute the average active signal over time.
- ``discrete_signal(signal0, stepSize)``
    Transform continuous signals into discrete steps.
- ``get_signal(events, stepSize, prob, pred, numClasses, **kargs)``
    Derive trade signals from classifiers and probabilities.
- ``betSize(x, w)``
    Calculate the bet size using a sigmoid function.
- ``getTargetPos(w, f, mP, maxPos)``
    Compute the target position given forecast and current holdings.
- ``invPrice(f, w, m)``
    Inverse of the bet size function for calibration.
- ``limitPrice(tPos, pos, f, w, maxPos)``
    Breakeven limit price for a target position change.
- ``getW(x, m)``
    Calibrate the sigmoid coefficient.
- ``getNumConcBets(date, signals, freq='B')``
    Count concurrent bets in a time window.
- ``getBetsTiming(tPos)``
    Retrieve the start and end times of bets.
- ``getHoldingPeriod(tPos)``
    Compute holding period for each bet.
- ``getHHI(betRet)``
    Herfindahl-Hirschman concentration index of bet returns.
- ``computeDD_TuW(series, dollars=False)``
    Drawdown and time-under-water calculations.
- ``Batch(coeffs, ...)`` and ``processBatch``
    Simulate betting paths for parameter analysis.
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

def avgActiveSignals_(signals: pd.DataFrame, molecule: np.ndarray):
    '''
    Auxilary function for averaging signals. At time loc, averages signal among those still active.
    Signal is active if:
        a) issued before or at loc AND
        b) loc before signal's endtime, or endtime is still unknown (NaT).

        Parameters:
            signals (pd.DataFrame): dataset with signals and t1
            molecule (np.ndarray): dates of events on which weights are computed

        Returns:
            out (pd.Series): series with average signals for each timestamp
    '''
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0  # no signals active at this time
    return out


def avgActiveSignals(signals: pd.DataFrame):
    '''
    Computes the average signal among those active.

        Parameters:
            signals (pd.DataFrame): dataset with signals and t1

        Returns:
            out (pd.Series): series with average signals for each timestamp
    '''
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = sorted(list(tPnts))
    out = avg_active_signals_(signals=signals, molecule=tPnts)
    return out

def discrete_signal(signal0: pd.Series, stepSize: float):
    '''
    Discretizes signals.

        Parameters:
            signal0 (pd.Series): series with signals
            stepSize (float): degree of discretization (must be in (0, 1])

        Returns:
            signal1 (pd.Series): series with discretized signals
    '''
    signal1 = (signal0 / stepSize).round() * stepSize  # discretize
    signal1[signal1 > 1] = 1  # cap
    signal1[signal1 < -1] = -1  # floor
    return signal1


def get_signal(events: pd.DataFrame, stepSize: float, prob: pd.Series, pred: pd.Series, numClasses: int, **kargs):
    '''
    Gets signals from predictions. Includes averaging of active bets as well as discretizing final value.

        Parameters:
            events (pd.DataFrame): dataframe with columns:
                                       - t1: timestamp of the first barrier touch
                                       - trgt: target that was used to generate the horizontal barriers
                                       - side (optional): side of bets
            stepSize (float): ---
            prob (pd.Series): series with probabilities of given predictions
            pred (pd.Series): series with predictions
            numClasses (int): number of classes

        Returns:
            signal1 (pd.Series): series with discretized signals
    '''
    if prob.shape[0] == 0:
        return pd.Series()
    signal0 = (prob - 1.0 / numClasses) / (prob * (1.0 - prob)) ** 0.5  # t-value
    signal0 = pred * (2 * norm.cdf(signal0) - 1)  # signal = side * size
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']  # meta-labeling
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avg_active_signals(df0)
    signal1 = discrete_signal(signal0=df0, stepSize=stepSize)
    return signal1


def betSize(x: float, w: float) -> float:
    '''
    Returns bet size given price divergence and sigmoid function coefficient.

        Parameters:
            x (float): difference between forecast price and current price f_i - p_t
            w (float): coefficient that regulates the width of the sigmoid function

        Returns:
            (float): bet size
    '''
    return x * (w + x ** 2) ** (-0.5)


def getTargetPos(w: float, f: float, mP: float, maxPos: float) -> float:
    '''
    Calculates target position size associated with forecast f.

        Parameters:
            w (float): coefficient that regulates the width of the sigmoid function
            f (float): forecast price
            mP (float): current market price
            maxPos (float): maximum absolute position size

        Returns:
            (float): target position size
    '''
    return int(bet_size(w, f - mP) * maxPos)


def invPrice(f: float, w: float, m: float) -> float:
    '''
    Calculates inverse function of bet size with respect to market price p_t.

        Parameters:
            f (float): forecast price
            w (float): coefficient that regulates the width of the sigmoid function
            m (float): bet size

        Returns:
            (float): inverse price function
    '''
    return f - m * (w / (1 - m ** 2)) ** 0.5


def limitPrice(tPos: float, pos: float, f: float, w: float, maxPos: float) -> float:
    '''
    Calculates the breakeven limit price ``p_bar`` for an order of size
    ``q_hat_{i,t} - q_t`` so that no losses are realised.

        Parameters:
            tPos (float): target position
            pos (float): current position
            f (float): forecast price
            w (float): coefficient that regulates the width of the sigmoid function
            maxPos (float): maximum absolute position size

        Returns:
            lP (float): limit price
    '''
    sgn = (1 if tPos >= pos else -1)
    lP = 0
    for j in range(abs(pos + sgn), abs(tPos + 1)):
        lP += invPrice(f, w, j / float(maxPos))
    lP /= tPos - pos
    return lP


def getW(x: float, m: float):
    '''
    Calibrates sigmoid coefficient by calculating the inverse function of bet size with respect to w.

        Parameters:
            x (float): difference between forecast price and current price f_i - p_t
            m (float): bet size
    '''
    return x ** 2 * (m ** (-2) - 1)


def getNumConcBets(date, signals, freq = 'B'):
    '''
    Derives number of long and short concurrent bets by given date.

        Parameters:
            date (Timestamp): date of signal
            signals (pd.DataFrame): dataframe with signals

        Returns:
            long, short (Tuple[int, int]): number of long and short concurrent bets
    '''
    long, short = 0, 0
    for ind in pd.date_range(start = max(signals.index[0], date - timedelta(days = 25)), end = date, freq = freq):
        if ind <= date and signals.loc[ind]['t1'] >= date:
            if signals.loc[ind]['signal'] >= 0:
                long += 1
            else:
                short += 1
    return long, short

# =================================================================================================================
#      Backtest Statistics
# =================================================================================================================

def getBetsTiming(tPos: pd.Series):
    df0 = tPos[tPos == 0].index
    df1 = tPos.shift(1)
    df1 = df1[df1 != 0].index
    bets = df0.intersection(df1)  # flattening
    df0 = tPos.iloc[1:] * tPos.iloc[:-1].values
    bets = bets.union(df0[df0 < 0].index).sort_values()  # tPos flips
    if tPos.index[-1] not in bets:
        bets = bets.append(tPos.index[-1:])  # last bet
    return bets


def getHoldingPeriod(tPos: pd.Series):
    hp, tEntry = pd.DataFrame(columns=['dT', 'w']), 0.0
    pDiff, tDiff = tPos.diff(), (tPos.index - tPos.index[0]) / np.timedelta64(1, 'D')
    for i in range(1, tPos.shape[0]):
        if pDiff.iloc[i] * tPos.iloc[i - 1] >= 0:  # increased or unchanged
            if tPos.iloc[i] != 0:
                tEntry = (tEntry * tPos.iloc[i - 1] + tDiff[i] * pDiff.iloc[i]) / tPos.iloc[i]
        else:  # decreased
            if tPos.iloc[i] * tPos.iloc[i - 1] < 0:  # flip
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(tPos.iloc[i - 1]))
                tEntry = tDiff[i]  # reset entry time
            else:
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(pDiff.iloc[i]))
    if hp['w'].sum() > 0:
        hp = (hp['dT'] * hp['w']).sum() / hp['w'].sum()
    else:
        hp = np.nan
    return hp


def getHHI(betRet: pd.Series):
    '''
    Derives HHI concentration of returns (see p. 200 for definition). Returns can be divided into positive
    and negative or you can calculate the concentration of bets across the months.

    Parameters:
        betRet (pd.Series): series with bets returns

    Returns:
        hhi (float): concentration
    '''
    if betRet.shape[0] <= 2:
        return np.nan
    wght = betRet / betRet.sum()
    hhi = (wght ** 2).sum()
    hhi = (hhi - betRet.shape[0] ** (-1)) / (1.0 - betRet.shape[0] ** (-1))
    return hhi


def computeDD_TuW(series: pd.Series, dollars: bool = False):
    '''
    Compute the drawdown and time underwater for a return series.

    Parameters:
        series (pd.Series): Price or sampled bar series.
        dollars (bool): Whether ``series`` represents dollars rather than
            returns.

    Returns:
        tuple(pd.Series, pd.Series): drawdown and time-under-water series.
    '''
    df0 = series.to_frame('pnl')
    df0['hwm'] = series.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = df0['hwm'].drop_duplicates(keep='first').index  # time of hwm
    df1 = df1[df1['hwm'] > df1['min']]  # hwm followed by a drawdown
    if dollars:
        dd = df1['hwm'] - df1['min']
    else:
        dd = 1 - df1['min'] / df1['hwm']
    tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, 'Y')).values  # in years
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw

def Batch(coeffs, nIter = 1e5, maxHP = 100, rPT = np.linspace(.5, 10, 20), rSLm = np.linspace(.5, 10, 20), seed = 42) :
    phi, output1 = 2 ** (-1 / coeffs['hl']), []
    for comb_ in product(rPT, rSLm) :
        output2 = []
        for iter_ in range(int(nIter)) :
            p, hp, count = seed, 0, 0
            while True :
                p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
                cP = p - seed
                hp += 1
                if cP > comb_[0] or cP < -comb_[1] or hp > maxHP :
                    output2.append(cP)
                    break
        mean, std = np.mean(output2), np.std(output2)
        print(comb_[0], comb_[1], mean, std, mean/std)
        output1.append((comb_[0], comb_[1], mean, std, mean/std))
    return output1

def processBatch(coeffs_list, **kwargs):
    out = []
    for coeffs in coeffs_list:
        out.append((coeffs, Batch(coeffs, **kwargs)))
    return out

__all__ = ['avgActiveSignals_','avgActiveSignals','discrete_signal','get_signal','betSize','getTargetPos','invPrice','limitPrice','getW','getNumConcBets','getBetsTiming','getHoldingPeriod','getHHI','computeDD_TuW','Batch','processBatch']
