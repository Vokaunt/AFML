"""High-frequency data simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HighFrequencyDataSimulator:
    """Generate synthetic trade data."""

    seed: int | None = None

    def generate_trades(self, n_trades: int, initial_price: float = 100.0) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        price_changes = rng.normal(0, 0.01, size=n_trades)
        prices = initial_price + np.cumsum(price_changes)
        volumes = rng.integers(1, 100, size=n_trades)
        timestamps = pd.date_range(start="2023-01-01", periods=n_trades, freq="s")
        return pd.DataFrame({"price": prices, "volume": volumes}, index=timestamps)
