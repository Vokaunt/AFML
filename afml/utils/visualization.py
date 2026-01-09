"""Visualization utilities for AFML examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class VisualizationTools:
    """Convenience plotting utilities for AFML workflows."""

    @staticmethod
    def plot_portfolio_performance(results: pd.DataFrame):
        """Plot cumulative returns from a backtest results DataFrame."""
        fig, ax = plt.subplots(figsize=(10, 4))
        if "returns" in results:
            cumulative = (1 + results["returns"]).cumprod()
            ax.plot(cumulative.index, cumulative.values, label="Cumulative Return")
        elif "equity" in results:
            ax.plot(results.index, results["equity"], label="Equity")
        ax.set_title("Portfolio Performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    @staticmethod
    def plot_weight_history(weight_history: pd.DataFrame):
        """Plot portfolio weights through time."""
        fig, ax = plt.subplots(figsize=(10, 4))
        weight_history.plot.area(ax=ax, stacked=True, alpha=0.8)
        ax.set_title("Portfolio Weight History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        return fig

    @staticmethod
    def plot_efficient_frontier(frontier_df: pd.DataFrame):
        """Plot risk vs return for an efficient frontier."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(frontier_df["volatility"], frontier_df["return"], marker="o")
        ax.set_title("Efficient Frontier")
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        ax.grid(True, alpha=0.3)
        return fig

    @staticmethod
    def plot_minimum_track_record_length(
        sharpe_ratio: float, target_sharpe_ratios: Iterable[float]
    ):
        """Plot minimum track record length for target Sharpe ratios."""
        fig, ax = plt.subplots(figsize=(6, 4))
        target_sharpe_ratios = list(target_sharpe_ratios)
        trl = [
            1 + ((s - sharpe_ratio) ** 2) * 252 for s in target_sharpe_ratios
        ]
        ax.plot(target_sharpe_ratios, trl, marker="o")
        ax.set_title("Minimum Track Record Length")
        ax.set_xlabel("Target Sharpe")
        ax.set_ylabel("Length (days)")
        ax.grid(True, alpha=0.3)
        return fig

    @staticmethod
    def plot_deflated_sharpe_ratio(
        sharpe_ratio: float, n_trials_range: Iterable[int], n_obs: int
    ):
        """Plot deflated Sharpe ratio across trial counts."""
        fig, ax = plt.subplots(figsize=(6, 4))
        n_trials_range = np.array(list(n_trials_range))
        dsr = sharpe_ratio / np.sqrt(1 + np.log(n_trials_range))
        ax.plot(n_trials_range, dsr, marker="o")
        ax.set_title("Deflated Sharpe Ratio")
        ax.set_xlabel("Number of Trials")
        ax.set_ylabel("Deflated Sharpe")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        return fig

    @staticmethod
    def plot_stochastic_dominance(
        returns1: np.ndarray,
        returns2: np.ndarray,
        label1: str,
        label2: str,
    ):
        """Plot CDFs for stochastic dominance comparison."""
        fig, ax = plt.subplots(figsize=(6, 4))
        for data, label in [(returns1, label1), (returns2, label2)]:
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf, label=label)
        ax.set_title("Stochastic Dominance")
        ax.set_xlabel("Return")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    @staticmethod
    def plot_cusum(cusum_stats: pd.DataFrame, threshold: float, title: str):
        """Plot CUSUM statistics with threshold bands."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(cusum_stats.index, cusum_stats["S+"], label="S+")
        ax.plot(cusum_stats.index, cusum_stats["S-"], label="S-")
        ax.axhline(threshold, color="red", linestyle="--")
        ax.axhline(-threshold, color="red", linestyle="--")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
