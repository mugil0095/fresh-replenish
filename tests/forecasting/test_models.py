import numpy as np
import pytest

from fresh_replenish.forecasting.models import lightgbm_quantiles, seasonal_naive


def test_seasonal_naive_repeats_last_week_as_the_median(make_series):
    s = make_series(np.arange(28, dtype=float).reshape(2, 14))
    q = seasonal_naive(s, origin=10, horizon=7)
    assert q.shape == (2, 9, 7)
    assert q[0, 4].tolist() == [3, 4, 5, 6, 7, 8, 9]  # days 3-9 for days 10-16
    assert np.isnan(np.delete(q, 4, axis=1)).all()  # a point forecast has no other deciles


def test_seasonal_naive_does_not_peek(make_series, weekly, no_peeking):
    no_peeking(seasonal_naive, make_series(weekly()), origin=56)


def test_lightgbm_learns_promotions_that_seasonal_naive_misses(make_series, weekly):
    promo = np.random.default_rng(1).random((40, 63)) < 0.2
    inputs = np.zeros((40, 63, 3), np.float32)
    inputs[..., 1] = promo  # activity_flag
    s = make_series(weekly(n=40) * np.where(promo, 2.0, 1.0), inputs)
    actual = s.demand[:, 56:]

    def wape(q):
        return np.abs(q[:, 4] - actual).sum() / actual.sum()

    q = lightgbm_quantiles(s, origin=56, horizon=7)
    assert q.shape == (40, 9, 7)
    assert (np.diff(q, axis=1) >= 0).all()  # deciles never decrease
    assert wape(q) < wape(seasonal_naive(s, 56, 7))


def test_lightgbm_does_not_peek(make_series, weekly, no_peeking):
    no_peeking(lightgbm_quantiles, make_series(weekly(n=40)), origin=56)


def test_seasonal_naive_needs_a_week_of_history(make_series, weekly):
    with pytest.raises(ValueError, match="7 days"):
        seasonal_naive(make_series(weekly()), origin=3, horizon=7)
