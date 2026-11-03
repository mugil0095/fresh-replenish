import numpy as np
import pandas as pd
import pytest

from fresh_replenish.forecasting.data import ATTRIBUTES, Series


@pytest.fixture
def weekly():
    """weekly(n, days, seed): positive demand with a weekly pattern, a level per series, noise."""
    return _weekly


def _weekly(n: int = 20, days: int = 63, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = rng.uniform(2, 10, (n, 1))
    pattern = 1 + 0.5 * np.sin(2 * np.pi * np.arange(days) / 7)
    return level * pattern * rng.gamma(20, 1 / 20, (n, days))


@pytest.fixture
def make_series():
    """make_series(demand, inputs=None) -> Series with synthetic ids and dates."""

    def make(demand, inputs=None):
        n, days = demand.shape
        ids = pd.DataFrame({
            "store_id": np.arange(n) // 2,
            "product_id": np.arange(n) % 2 + 10,
            **{c: np.zeros(n, dtype=int) for c in ATTRIBUTES},
        })
        return Series(
            ids=ids,
            dates=np.arange(days).astype(str),
            demand=demand.astype(np.float32),
            in_stock=np.ones((n, days), dtype=bool),
            inputs=np.zeros((n, days, 3), np.float32) if inputs is None else inputs,
        )

    return make


@pytest.fixture
def no_peeking():
    """no_peeking(forecaster, series, origin): demand changed from the origin on changes nothing."""
    return _no_peeking


def _no_peeking(forecaster, series, origin, horizon=7):
    before = forecaster(series, origin, horizon)
    series.demand[:, origin:] *= 3
    assert np.array_equal(before, forecaster(series, origin, horizon), equal_nan=True)
