"""Forecasters: each maps (series, origin, horizon) to deciles of shape (series, 9, horizon).

A forecaster reads demand only before `origin`, plus the known inputs on the forecast days.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from fresh_replenish.forecasting.data import ATTRIBUTES, HORIZON, INPUTS, Series
from fresh_replenish.metrics import DECILES

MEDIAN = len(DECILES) // 2
LAGS = (7, 14, 21, 28)  # every demand feature lags 7+ days, so one model covers the whole week
FEATURES = [
    *(f"lag{k}" for k in LAGS), "mean7", "mean28", "weekday", *INPUTS,
    "store_id", "product_id", *ATTRIBUTES,
]
CATEGORICAL = [FEATURES.index(c) for c in ["weekday", "store_id", "product_id", *ATTRIBUTES]]
LGBM = {
    "objective": "quantile", "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 63,
    "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": 42, "deterministic": True, "force_row_wise": True, "verbose": -1,
}


def _tidy(q: np.ndarray) -> np.ndarray:
    """Demand can't be negative, and deciles must not cross."""
    return np.sort(np.maximum(q, 0), axis=1).astype(np.float32)


def seasonal_naive(series: Series, origin: int, horizon: int = HORIZON) -> np.ndarray:
    """Same weekday last week; a point forecast, so only the median is filled."""
    q = np.full((len(series.demand), len(DECILES), horizon), np.nan, dtype=np.float32)
    q[:, MEDIAN] = series.demand[:, origin - 7 : origin][:, np.arange(horizon) % 7]
    return q


def _features(series: Series, origin: int, horizon: int) -> np.ndarray:
    """(series, origin + horizon days, FEATURES); demand enters only from before the origin."""
    n, days = len(series.demand), origin + horizon
    known = pd.DataFrame(series.demand[:, :origin].T).reindex(range(days))  # (days, series)
    demand = [known.shift(k) for k in LAGS]
    demand += [known.rolling(w).mean().shift(7) for w in (7, 28)]
    ids = series.ids[["store_id", "product_id", *ATTRIBUTES]].to_numpy()
    return np.concatenate([
        np.stack([frame.to_numpy().T for frame in demand], axis=-1),
        np.broadcast_to((np.arange(days) % 7)[None, :, None], (n, days, 1)),
        series.inputs[:, :days],
        np.broadcast_to(ids[:, None, :], (n, days, ids.shape[1])),
    ], axis=-1).astype(np.float32)


def lightgbm_quantiles(series: Series, origin: int, horizon: int = HORIZON) -> np.ndarray:
    """One global LightGBM per decile, trained on every series' days before the origin."""
    x = _features(series, origin, horizon)
    n, _, width = x.shape
    train_x, train_y = x[:, :origin].reshape(-1, width), series.demand[:, :origin].ravel()
    q = np.stack([
        lgb.LGBMRegressor(alpha=alpha, **LGBM)
        .fit(train_x, train_y, categorical_feature=CATEGORICAL)
        .predict(x[:, origin:].reshape(-1, width))
        for alpha in DECILES
    ], axis=-1)  # (series * horizon, 9)
    return _tidy(q.reshape(n, horizon, len(DECILES)).transpose(0, 2, 1))
