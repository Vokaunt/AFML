"""Hyperparameter search utilities.

Functions
---------
- ``MyPipeline(steps)``
    Light wrapper around ``sklearn`` pipeline.
- ``clfHyperFit(clf, param_grid, X, y, cv, t1, sample_weight)``
    Grid search using ``PurgedKFold``.
- ``clfHyperFitRand(clf, param_grid, X, y, cv, t1, sample_weight, n_iter)``
    Randomized search variant.
- ``logUniform(low, high, size=None)``
    Draw samples from a log-uniform distribution.
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
#      Hyper Parameter Tuning
# =================================================================================================================

class MyPipeline(Pipeline):
    """
    Inherit all methods from sklearn's `Pipeline`
    Overwrite the inherited `fit` method with a new one that handles the argument `sample weight`
    After which it redirects to the parent class
    """
    def fit(self, X, y, sample_weight = None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)

def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid,
                cv = 3, bagging = [0, None, 1.], n_jobs = -1, pctEmbargo = 0, **fit_params):
    """
    Grid Search with purged K-Fold Cross Validation
    :params feat: features
    :params lbl: labels
    :params t1: vertical barriers
    :params pipe_clf: classification pipeline
    :params param_grid: parameter grid
    :params cv: int, cross validation fold
    :params bagging: bagging parameter?
    :params n_jobs: CPUs
    :params pctEmbargo: float, % of embargo
    :params **fit_params:
    :return gs:
    """
    scoring = 'f1' # f1 for meta-labeling
    # if set(lbl.values) == {0, 1}: # if label values are 0 or 1
    #     scoring = 'f1' # f1 for meta-labeling
    # else:
    #     scoring = 'neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data
    # prepare the training sets and the validation sets for CV (find their indices)
    inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged
    # perform grid search
    gs = GridSearchCV(estimator = pipe_clf, param_grid = param_grid,
                        scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False)
    # best estimator and the best parameter
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # pipeline
    #2) fit validated model on the entirety of the data
    if bagging[1] > 0: # max_samples > 0
        gs = BaggingClassifier(base_estimator = MyPipeline(gs.steps),
                                n_estimators = int(bagging[0]), max_samples = float(bagging[1]),
                                max_features = float(bagging[2]), n_jobs = n_jobs)
        gs = gs.fit(feat, lbl, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs

def clfHyperFitRand(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0,None,1.], rndSearchIter=0, n_jobs=-1, pctEmbargo=0, **fit_params):
    """
    Randimised Search with Purged K-fold CV
    Grid Search with purged K-Fold Cross Validation
    :params feat: features
    :params lbl: labels
    :params t1: vertical barriers, used for PurgedKFold
    :params pipe_clf: classification pipeline
    :params param_grid: parameter grid
    :params cv: int, cross validation fold
    :params bagging: bagging parameter?
    :params rndSearchIter
    :params n_jobs: CPUs
    :params pctEmbargo: float, % of embargo
    :params **fit_params:
    :return gs:
    """
    if set(lbl.values) == {0,1}:# if label values are 0 or 1
        scoring = 'f1' # f1 for meta-labeling
    else:
        scoring = 'neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data
    # prepare the training sets and the validation sets for CV (find their indices)
    inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged
    if rndSearchIter == 0: # randomised grid search
        gs = GridSearchCV(estimator = pipe_clf, param_grid = param_grid,
                            scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False)
    else: # normal grid search
        gs = RandomizedSearchCV(estimator = pipe_clf, param_distributions = param_grid,
                                scoring = scoring, cv = inner_cv, n_jobs = n_jobs,
                                iid = False, n_iter = rndSearchIter)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # pipeline
    #2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator = MyPipeline(gs.steps),
                                n_estimators = int(bagging[0]), max_samples = float(bagging[1]),
                                max_features = float(bagging[2]), n_jobs = n_jobs)
        gs = gs.fit(feat, lbl, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs

from scipy.stats import rv_continuous,kstest
class logUniform_gen(rv_continuous):
# random numbers log-uniformly distributed between 1 and e
    def _cdf(self,x):
        return np.log(x/self.a)/np.log(self.b/self.a)
def logUniform(a = 1,b = np.exp(1)) : return logUniform_gen(a = a,b = b,name = 'logUniform')


__all__ = ['MyPipeline','clfHyperFit','clfHyperFitRand','logUniform']
