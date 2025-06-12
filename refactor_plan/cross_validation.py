"""Cross-validation utilities.

Functions and classes
---------------------
- ``getTrainTimes(t1: pd.Series, testTimes: pd.Series) -> pd.Series``
    Remove training observations whose label spans overlap the test set.

- ``getEmbargoTimes(times: pd.Index, pctEmbargo: float) -> pd.Series``
    Create an embargo period following each test observation to avoid leakage.

- ``PurgedKFold``
    A ``KFold`` variant that drops training samples overlapping in time with the
    test fold and optionally applies an embargo.

- ``cvScore(...) -> list``
    Run cross validation using ``PurgedKFold`` and return scores per fold.

- ``crossValPlot(results: list, ax: matplotlib.axes.Axes)``
    Plot ROC curves across folds and show the mean performance.
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
#           Cross Validation
# =================================================================================================================
def getTrainTimes(t1, testTimes):
    """
    Given testTimes, find the times of the training observations.
    Purge from the training set all observations whose labels overlapped in time with those labels included in the testing set
    :param t1: pd.Series of event start and end times
        - ``t1.index``: observation start time
        - ``t1.value``: observation end time
    :param testTimes: pd.Series with test observation intervals
    :return: pd.Series of training timestamps with overlaps removed
    """
    # copy t1 to trn
    trn = t1.copy(deep=True)
    # for every times of testing obervation
    for i, j in testTimes.iteritems():
        # cond 1: train starts within test
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        # cond 2: train ends within test
        df1 = trn[(i <= trn) & (trn <= j)].index
        # cond 3: train envelops test
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        # drop the data that satisfy cond 1 & 2 & 3
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getEmbargoTimes(times, pctEmbargo):
     """ Not sure if it works
     # Get embargo time for each bar
     :params times: time bars
     :params pctEmbargo: float, % of the bars will be embargoed
     :return trn: pd.df, purged training set
     """
     # cal no. of steps from the test data
     step = int(times.shape[0] * pctEmbargo)
     if step == 0:
         # if no embargo, the same data set
         mbrg=pd.Series(times,index=times)
     else:
         #
         mbrg=pd.Series(times[step:],index=times[:-step])
         mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:]))
     return mbrg

class PurgedKFold(_BaseKFold):
    """KFold variant that purges training observations overlapping the test set.

    It assumes the test set is contiguous and that no shuffle is applied.
    Overlapping samples are dropped and an optional embargo can be added
    between folds.
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            # if t1 is not a pd.series, raise error
            raise ValueError('Label Through Dates must be a pd.Series')
        # inherit _BaseKFold, no shuffle
        # Might be python 2x style
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1  # specify the vertical barrier
        self.pctEmbargo = pctEmbargo  # specify the embargo parameter (% of the bars)

    def split(self, X, y=None, groups=None):
        """
        :param X: the regressors, features
        :param y: the regressands, labels
        :param groups: None

        : return
            + train_indices: generator, the indices of training dataset
            + test_indices: generator, the indices of the testing dataset
        """
        if (pd.DataFrame(X).index == self.t1.index).sum() != len(self.t1):
            # X's index does not match t1's index, raise error
            raise ValueError('X and ThruDateValues must have the same index')
        # create an array from 0 to (X.shape[0]-1)
        indices = np.arange(X.shape[0])
        # the size of the embargo
        mbrg = int(X.shape[0] * self.pctEmbargo)
        # list comprehension, find the (first date, the last date + 1) of each split
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:  # for each split
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i: j]  # test indices are all the indices from i to j
            maxT1Idx = self.t1.index.searchsorted(
                self.t1[test_indices].max())  # find the max(furthest) vertical barrier among the test dates
            # index.searchsorted: find indices where element should be inserted (behind) to maintain the order
            # find all t1.indices (the start dates of the event) when t1.value (end date) < t0
            # i.e the left side of the training data
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                # indices[maxT1Idx+mbrg:]: the indices that is after the (maxTestDate + embargo period) [right training set]
                # concat the left training indices and the right training indices
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
        # the function return generators for the indices of training dataset and the indices of the testing dataset respectively
        yield train_indices, test_indices

def cvScore(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=0):
    """
    Address two sklearn bugs
    1) Scoring functions do not know classes_
    2) cross_val_score will give different results because it weights to the fit method, but not to the log_loss method

    :params pctEmbargo: float, % of the bars will be embargoed
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        # if not using 'neg_log_loss' or 'accuracy' to score, raise error
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score  # import log_loss and accuracy_score
    if cvGen is None:  # if there is no predetermined splits of the test sets and the training sets
        # use the PurgedKFold to generate splits of the test sets and the training sets
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    score = []  # store the CV scores
    # for each fold
    for train, test in cvGen.split(X = X):
        # fit the model
        fit = clf.fit(X = pd.DataFrame(X).iloc[train, :], y = pd.DataFrame(y).iloc[train],
                      sample_weight = pd.DataFrame(sample_weight).iloc[train].values.reshape(1,-1)[0])
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(pd.DataFrame(X).iloc[test, :])  # predict the probabily
            # neg log loss to evaluate the score
            score_ = -1 * log_loss(pd.DataFrame(y).iloc[test], prob,
                                   sample_weight = pd.DataFrame(sample_weight).iloc[test].values.reshape(1,-1)[0],
                                   labels=clf.classes_)
        else:
            pred = fit.predict(pd.DataFrame(X).iloc[test, :])  # predict the label
            # predict the accuracy score
            score_ = accuracy_score(pd.DataFrame(y).iloc[test], pred,
                                    sample_weight = pd.DataFrame(sample_weight).iloc[test].values.reshape(1, -1)[0])
        score.append(score_)
    return np.array(score)

def crossValPlot(skf,classifier,X_,y_):
    """Code adapted from:

    """

    X = pd.DataFrame(X_)
    X = np.asarray(X)
    y = np.asarray(y_)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    f,ax = plt.subplots(figsize=(10,7))
    i = 0
    for train, test in skf.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(bbox_to_anchor=(1,1))

__all__ = ['PurgedKFold','cvScore','crossValPlot','getTrainTimes','getEmbargoTimes']
