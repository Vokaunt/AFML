"""Bar sampling utilities."""

from __future__ import annotations

import pandas as pd


class BarSampler:
    """Generate different bar types from tick data."""

    def __init__(self, tick_data: pd.DataFrame):
        self.tick_data = tick_data.sort_index()

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.resample("1min").agg({"price": "last", "volume": "sum"}).dropna()

    def get_time_bars(self, freq: str) -> pd.DataFrame:
        return self.tick_data.resample(freq).agg({"price": "last", "volume": "sum"}).dropna()

    def get_tick_bars(self, tick_threshold: int) -> pd.DataFrame:
        groups = self.tick_data.groupby(self.tick_data.index.floor("1s")).head(tick_threshold)
        return self._aggregate(groups)

    def get_volume_bars(self, volume_threshold: int) -> pd.DataFrame:
        cum_volume = self.tick_data["volume"].cumsum()
        bins = (cum_volume // volume_threshold).astype(int)
        return self.tick_data.groupby(bins).agg({"price": "last", "volume": "sum"})

    def get_dollar_bars(self, dollar_threshold: float) -> pd.DataFrame:
        dollar = (self.tick_data["price"] * self.tick_data["volume"]).cumsum()
        bins = (dollar // dollar_threshold).astype(int)
        return self.tick_data.groupby(bins).agg({"price": "last", "volume": "sum"})

    def get_tick_imbalance_bars(self, expected_ticks_per_bar: int) -> pd.DataFrame:
        return self.get_tick_bars(expected_ticks_per_bar)

    def get_volume_imbalance_bars(self, expected_volume_per_bar: int) -> pd.DataFrame:
        return self.get_volume_bars(expected_volume_per_bar)

    def get_dollar_imbalance_bars(self, expected_dollar_per_bar: float) -> pd.DataFrame:
        return self.get_dollar_bars(expected_dollar_per_bar)

    def get_tick_runs_bars(self) -> pd.DataFrame:
        return self.get_tick_bars(100)

    def get_volume_runs_bars(self) -> pd.DataFrame:
        return self.get_volume_bars(50_000)

    def get_dollar_runs_bars(self) -> pd.DataFrame:
        return self.get_dollar_bars(1_000_000)
