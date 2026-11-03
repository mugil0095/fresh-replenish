import numpy as np

from fresh_replenish.forecasting.models import seasonal_naive


def test_seasonal_naive_repeats_last_week_as_the_median(make_series):
    s = make_series(np.arange(28, dtype=float).reshape(2, 14))
    q = seasonal_naive(s, origin=10, horizon=7)
    assert q.shape == (2, 9, 7)
    assert q[0, 4].tolist() == [3, 4, 5, 6, 7, 8, 9]  # days 3-9 for days 10-16
    assert np.isnan(np.delete(q, 4, axis=1)).all()  # a point forecast has no other deciles


def test_seasonal_naive_does_not_peek(make_series, weekly, no_peeking):
    no_peeking(seasonal_naive, make_series(weekly()), origin=56)
