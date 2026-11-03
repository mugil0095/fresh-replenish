"""Forecasters: each maps (series, origin, horizon) to deciles of shape (series, 9, horizon).

A forecaster reads demand only before `origin`, plus the known inputs on the forecast days.
"""

import numpy as np

from fresh_replenish.forecasting.data import HORIZON, Series
from fresh_replenish.metrics import DECILES

MEDIAN = len(DECILES) // 2


def seasonal_naive(series: Series, origin: int, horizon: int = HORIZON) -> np.ndarray:
    """Same weekday last week; a point forecast, so only the median is filled."""
    q = np.full((len(series.demand), len(DECILES), horizon), np.nan, dtype=np.float32)
    q[:, MEDIAN] = series.demand[:, origin - 7 : origin][:, np.arange(horizon) % 7]
    return q
