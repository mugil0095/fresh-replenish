import numpy as np
import pytest

from fresh_replenish.imputation.statistical import (
    expected_above,
    fit_tobit,
    profile_recover,
)

PROFILE = np.linspace(1, 2, 16) / np.linspace(1, 2, 16).sum()


def known_sales(totals: np.ndarray) -> np.ndarray:
    """Hourly sales that follow PROFILE exactly: (series, days, 16)."""
    return totals[..., None] * PROFILE


def test_profile_recovers_partly_hidden_day_exactly():
    totals = np.array([[10.0, 20.0, 30.0], [4.0, 6.0, 8.0]])
    sales = known_sales(totals)
    sales[0, 1, 10:] = np.nan  # sold out from 16:00
    sales[1, 2, :5] = np.nan  # empty until 11:00
    filled = profile_recover(sales)
    assert filled.sum(axis=-1) == pytest.approx(totals)


def test_series_without_full_day_uses_pooled_profile():
    sales = known_sales(np.array([[10.0, 20.0], [5.0, 5.0]]))
    sales[1, :, 15] = np.nan  # series 1 never has a fully observed day
    filled = profile_recover(sales)
    assert filled[1, 0, 15] == pytest.approx(5.0 * PROFILE[15])


def test_fully_hidden_day_uses_series_mean_day():
    sales = known_sales(np.array([[10.0, 20.0, 0.0]]))
    sales[0, 2] = np.nan
    assert profile_recover(sales)[0, 2].sum() == pytest.approx(15.0)


def test_expected_above_matches_monte_carlo():
    draws = np.random.default_rng(0).normal(5.0, 2.0, 2_000_000)
    assert expected_above(5.0, 2.0, 6.0) == pytest.approx(draws[draws > 6.0].mean(), rel=0.02)


def test_tobit_recovers_slopes_and_spread():
    rng = np.random.default_rng(1)
    n, days, beta = 200, 60, np.array([0.8, -0.5])
    series = np.repeat(np.arange(n), days)
    X = rng.normal(size=(n * days, 2))
    sigma = rng.uniform(0.5, 1.5, n)
    demand = rng.uniform(2, 6, n)[series] + X @ beta + sigma[series] * rng.normal(size=n * days)
    censored = rng.random(n * days) < 0.3
    y = np.where(censored, demand * rng.uniform(0.3, 0.9, n * days), demand)  # stockout cuts sales
    fit = fit_tobit(y, censored, X, series, n)
    assert fit.beta == pytest.approx(beta, abs=0.1)
    assert np.median(np.abs(np.exp(fit.log_sigma) / sigma - 1)) < 0.1
