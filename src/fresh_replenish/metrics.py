"""Error measures shared by the phases, with bootstrap intervals over series."""

import numpy as np

SEED = 42
DECILES = np.round(np.arange(1, 10) / 10, 1)  # 0.1 ... 0.9


def score(pred, actual, valid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-series sums over valid days of |error|, error and actual demand."""
    err = np.where(valid, pred - actual, 0.0)
    return np.abs(err).sum(axis=1), err.sum(axis=1), np.where(valid, actual, 0.0).sum(axis=1)


def bootstrap(sums, n: int = 1000, seed: int = SEED) -> tuple[dict, np.ndarray]:
    """WAPE and WPE with 95% intervals from resampling series; also returns the WAPE draws."""
    abs_err, err, actual = sums
    k = len(actual)
    weights = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=n)
    wape = weights @ abs_err / (weights @ actual)
    wpe = weights @ err / (weights @ actual)
    point = {
        "wape": abs_err.sum() / actual.sum(),
        "wape_low": np.percentile(wape, 2.5),
        "wape_high": np.percentile(wape, 97.5),
        "wpe": err.sum() / actual.sum(),
        "wpe_low": np.percentile(wpe, 2.5),
        "wpe_high": np.percentile(wpe, 97.5),
    }
    return point, wape


def wql(q, actual) -> float:
    """Scaled pinball loss of deciles q (series, 9, days): mean of 2 * sum pinball / sum actual."""
    diff = actual[:, None, :] - q
    tau = DECILES[:, None]
    loss = np.maximum(tau * diff, (tau - 1) * diff).sum(axis=(0, 2))
    return float(2 * loss.mean() / actual.sum())


def coverage(q, actual) -> float:
    """Share of actuals inside [q0.1, q0.9]; NaN for a point forecast."""
    low, high = q[:, 0], q[:, -1]
    if np.isnan(low).any():
        return float("nan")
    return float(((actual >= low) & (actual <= high)).mean())


def mase(pred, actual, history, season: int = 7) -> np.ndarray:
    """Per-series MAE over the history's seasonal-naive MAE; zero-scale series are dropped."""
    scale = np.abs(history[:, season:] - history[:, :-season]).mean(axis=1)
    keep = scale > 0
    return np.abs(pred - actual).mean(axis=1)[keep] / scale[keep]
