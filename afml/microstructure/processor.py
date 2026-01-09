"""Tick data processing utilities."""

from __future__ import annotations

import pandas as pd


class TickDataProcessor:
    """Clean and aggregate tick data."""

    def clean_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        return trades.dropna().sort_index()

    def create_tick_bars(self, trades: pd.DataFrame, ticks_per_bar: int = 500) -> pd.DataFrame:
        groups = trades.groupby(trades.index.floor("1s"))
        sampled = groups.head(ticks_per_bar)
        return sampled.resample("1min").agg({"price": "last", "volume": "sum"}).dropna()
