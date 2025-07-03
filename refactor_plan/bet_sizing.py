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
from dataclasses import dataclass
from typing import Optional
import brownian_motion

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
from sklearn.linear_model import LinearRegression

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

# ======================================================================
# Additions from missing functions list
# ======================================================================

def expected_max(N: int) -> float:
    """Return expected maximum of ``N`` standard normal draws."""
    if N < 5:
        raise AssertionError("Condition N >> 1 not satisfied.")
    return (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / N)
        + np.euler_gamma * stats.norm.ppf(1 - np.exp(-1) / N)
    )


def PSR(sharpe: float, T: int, skew: float, kurtosis: float, target_sharpe: float = 0) -> float:
    """Probabilistic Sharpe Ratio adjusting for skew and kurtosis."""
    value = (
        (sharpe - target_sharpe)
        * np.sqrt(T - 1)
        / np.sqrt(1.0 - skew * sharpe + sharpe ** 2 * (kurtosis - 1) / 4.0)
    )
    return stats.norm.cdf(value, 0, 1)


def DSR(test_sharpe: float, sharpe_std: float, N: int, T: int, skew: float, kurtosis: float) -> float:
    """Compute Deflated Sharpe Ratio for given performance series."""
    target_sharpe = sharpe_std * expected_max(N)
    return PSR(test_sharpe, T, skew, kurtosis, target_sharpe)


def betSizePower(x: float, w: float) -> float:
    """Power-transformed bet size used as an alternative to ``betSize``."""
    return np.sign(w) * abs(w) ** x


def binSR(sl: float, pt: float, freq: float, p: float) -> float:
    """Theoretical annualized Sharpe ratio under a Bernoulli trading model."""
    mean = p * pt - (1 - p) * sl
    var = p * pt ** 2 + (1 - p) * sl ** 2 - mean ** 2
    return np.sqrt(freq) * mean / np.sqrt(var)


def binHR(sl: float, pt: float, freq: float, tSR: float) -> float:
    """Solve for hit rate ``p`` that achieves target Sharpe ratio ``tSR``."""
    a = (freq + tSR ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - tSR ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2.0 * a)


def binFreq(sl: float, pt: float, p: float, tSR: float) -> Optional[float]:
    """Frequency needed to reach a target Sharpe ratio."""
    freq = (tSR * (pt - sl)) ** 2 * p * (1 - p) / ((pt - sl) * p + sl) ** 2
    if not np.isclose(binSR(sl, pt, freq, p), tSR):
        return None
    return freq


def mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs):
    """Sample ``nObs`` observations from a two-component Gaussian mixture."""
    ret1 = np.random.normal(mu1, sigma1, size=int(nObs * prob1))
    ret2 = np.random.normal(mu2, sigma2, size=int(nObs) - ret1.shape[0])
    ret = np.append(ret1, ret2, axis=0)
    np.random.shuffle(ret)
    return ret


def probFailure(ret: np.ndarray, freq: float, tSR: float) -> float:
    """Probability that the Sharpe ratio does not exceed ``tSR``."""
    rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    thresP = binHR(rNeg, rPos, freq, tSR)
    return stats.norm.cdf(thresP, p, p * (1 - p))


def runSRtrials(p: float, pt: float = 1, sl: float = 1, trials: int = 100000) -> float:
    """Monte Carlo estimate of the Sharpe ratio for a Bernoulli strategy."""
    out = []
    for _ in range(trials):
        rnd = np.random.binomial(n=1, p=p)
        out.append(pt if rnd == 1 else -sl)
    return np.mean(out) / np.std(out)


def jiggle(v: float):
    """Return 1% variations around ``v``."""
    return [v * 0.99, v, v * 1.01]


def genHeatmap(forecast: float, hl: float, sigma: float = 1):
    """Run OU simulations and return batch output for heatmap plotting."""
    rPT = rSLm = np.linspace(0, 10, 21)
    coeffs = {"forecast": forecast, "hl": hl, "sigma": sigma}
    return Batch(coeffs, nIter=1e5, maxHP=100, rPT=rPT, rSLm=rSLm)


def OUcoeff():
    """Sweep OU parameters and run batch simulations."""
    rPT = rSLm = np.linspace(0, 10, 21)
    for prod_ in product([10, 5, 0, -5, -10], [5, 10, 25, 50, 100]):
        coeffs = {"forecast": prod_[0], "hl": prod_[1], "sigma": 1}
        output = Batch(coeffs, nIter=1e5, maxHP=100, rPT=rPT, rSLm=rSLm)
    return output


@dataclass
class OUParams:
    """Parameters of an Ornstein–Uhlenbeck process."""

    phi: float
    gamma: float
    sigma: float


def estimateOUParams(X_t: np.ndarray) -> OUParams:
    """Estimate OU parameters via ordinary least squares."""
    y = np.diff(X_t)
    X = X_t[:-1].reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    phi = -reg.coef_[0]
    gamma = reg.intercept_ / phi
    y_hat = reg.predict(X)
    sigma = np.std(y - y_hat)
    return OUParams(phi, gamma, sigma)


def getIntegalW(t: np.ndarray, dW: np.ndarray, OU_params: OUParams) -> np.ndarray:
    r"""Compute ``\int e^{phi t} dW`` for the OU process."""
    exp_phi_s = np.exp(OU_params.phi * t)
    integral_W = np.cumsum(exp_phi_s * dW)
    return np.insert(integral_W, 0, 0)[:-1]


def selectX0(X_0_in: Optional[float], OU_params: OUParams) -> float:
    """Choose initial value ``X_0``; defaults to long‑term mean."""
    return X_0_in if X_0_in is not None else OU_params.gamma


def getOUProcess(
    T: int,
    OU_params: OUParams,
    X_0: Optional[float] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate a sample path of an Ornstein–Uhlenbeck process."""
    t = np.arange(T, dtype=np.float128)
    exp_alpha_t = np.exp(-OU_params.phi * t)
    dW = brownian_motion.get_dW(T, random_state)
    integral_W = getIntegalW(t, dW, OU_params)
    _X_0 = selectX0(X_0, OU_params)
    return _X_0 * exp_alpha_t + OU_params.gamma * (1 - exp_alpha_t) + OU_params.sigma * exp_alpha_t * integral_W

__all__ = [
    'avgActiveSignals_','avgActiveSignals','discrete_signal','get_signal','betSize',
    'getTargetPos','invPrice','limitPrice','getW','getNumConcBets','getBetsTiming',
    'getHoldingPeriod','getHHI','computeDD_TuW','Batch','processBatch',
    'expected_max','PSR','DSR','betSizePower','binSR','binHR','binFreq',
    'mixGaussians','probFailure','runSRtrials','jiggle','genHeatmap','OUcoeff',
    'OUParams','estimateOUParams','getIntegalW','selectX0','getOUProcess'
]
