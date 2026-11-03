"""Statistical demand recovery: an in-stock-hour profile and a Tobit censored regression."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

MIN_COVER = 0.1  # below 10% of a day's usual sales observed, rescaling is too noisy
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def profile_recover(sales: np.ndarray) -> np.ndarray:
    """Fill blank hours from each series' in-stock hourly profile.

    sales: (series, days, hours) with NaN where demand is hidden. A partly blank day is scaled up
    by the share of the profile it observed; a (nearly) blank day uses the series' mean in-stock
    day. Series with no fully observed day borrow the pooled profile and mean.
    """
    blank = np.isnan(sales)
    observed = np.where(blank, 0.0, sales)
    full = ~blank.any(axis=-1)
    full_sums = np.where(full[..., None], observed, 0.0).sum(axis=1)  # (series, hours)
    totals = full_sums.sum(axis=1)
    n_full = full.sum(axis=1)

    pooled = full_sums.sum(axis=0) if full_sums.sum() > 0 else np.ones(sales.shape[-1])
    pooled_profile = pooled / pooled.sum()
    safe_totals = np.where(totals > 0, totals, 1.0)[:, None]
    profile = np.where((totals > 0)[:, None], full_sums / safe_totals, pooled_profile)
    pooled_mean = totals.sum() / max(n_full.sum(), 1)
    mean_day = np.where(n_full > 0, totals / np.maximum(n_full, 1), pooled_mean)

    covered = np.where(blank, 0.0, profile[:, None, :]).sum(axis=-1)
    day_total = np.where(
        covered >= MIN_COVER,
        observed.sum(axis=-1) / np.maximum(covered, MIN_COVER),
        mean_day[:, None],
    )
    return np.where(blank, profile[:, None, :] * day_total[..., None], sales)


def expected_above(mu, sigma, c):
    """E[D | D > c] for D ~ Normal(mu, sigma^2)."""
    z = (np.asarray(c) - mu) / sigma
    return mu + sigma * np.exp(-0.5 * z**2 - LOG_SQRT_2PI - log_ndtr(-z))


@dataclass
class TobitFit:
    alpha: np.ndarray  # (series,) intercepts
    log_sigma: np.ndarray  # (series,)
    beta: np.ndarray  # (covariates,) shared slopes


def fit_tobit(y, censored, X, series, n_series: int) -> TobitFit:
    """Right-censored normal regression: on censored rows the observed y is a lower bound.

    Demand ~ Normal(alpha[series] + X @ beta, exp(log_sigma[series])^2). A weak prior worth one
    pseudo-day per series pulls alpha and log_sigma toward the series' in-stock mean and spread,
    so a series that sold out every day still has a finite fit.
    """
    n, k = n_series, X.shape[1]
    free = ~censored
    count = np.bincount(series[free], minlength=n)
    alpha0 = np.where(count > 0, np.bincount(series[free], y[free], n) / np.maximum(count, 1),
                      np.bincount(series, y, n) / np.maximum(np.bincount(series, minlength=n), 1))
    spread = np.sqrt(np.maximum(
        np.bincount(series[free], y[free] ** 2, n) / np.maximum(count, 1) - alpha0**2, 0.0))
    sigma0 = np.maximum(np.where(count > 1, spread, np.std(y[free]) if free.any() else 1.0), 0.05)
    log_sigma0 = np.log(sigma0)

    def objective(theta):
        alpha, log_sigma, beta = theta[:n], theta[n : 2 * n], theta[2 * n :]
        s = np.exp(log_sigma)[series]
        z = (y - alpha[series] - X @ beta) / s
        log_sf = log_ndtr(-z)
        mills = np.exp(-0.5 * z**2 - LOG_SQRT_2PI - log_sf)
        ll = np.where(censored, log_sf, -0.5 * z**2 - np.log(s) - LOG_SQRT_2PI)
        d_mu = np.where(censored, mills / s, z / s)
        d_log_sigma = np.where(censored, mills * z, z**2 - 1)
        prior = (alpha - alpha0) / sigma0
        nll = -ll.sum() + 0.5 * (prior**2).sum() + 0.5 * ((log_sigma - log_sigma0) ** 2).sum()
        grad = np.concatenate([
            -np.bincount(series, d_mu, n) + prior / sigma0,
            -np.bincount(series, d_log_sigma, n) + (log_sigma - log_sigma0),
            -(X.T @ d_mu),
        ])
        return nll, grad

    theta0 = np.concatenate([alpha0, log_sigma0, np.zeros(k)])
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return TobitFit(res.x[:n], res.x[n : 2 * n], res.x[2 * n :])


def tobit_recover(sales: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Daily operating-hour demand (series, days): observed totals, or E[D | D > observed]."""
    n, days, _ = sales.shape
    blank = np.isnan(sales)
    y = np.where(blank, 0.0, sales).sum(axis=-1).ravel().astype(float)
    censored = blank.any(axis=-1).ravel()
    X = covariates.reshape(n * days, -1).astype(float)
    std = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    series = np.repeat(np.arange(n), days)
    fit = fit_tobit(y, censored, X, series, n)
    mu = fit.alpha[series] + X @ fit.beta
    sigma = np.exp(fit.log_sigma)[series]
    return np.where(censored, expected_above(mu, sigma, y), y).reshape(n, days)
