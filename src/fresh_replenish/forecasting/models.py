"""Forecasters: each maps (series, origin, horizon) to deciles of shape (series, 9, horizon).

A forecaster reads demand only before `origin`, plus the known inputs on the forecast days.
"""

import functools
import sys

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
# Foundation-model weights, pinned to the Hugging Face revisions downloaded on 2026-11-03.
CHRONOS2 = ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c")
TIMESFM25 = ("google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3")
TOTO2 = ("Datadog/Toto-2.0-313m", "a7bab288f5e95f8606f8306f86659357e1c001ef")
BATCH = 256  # series per forward pass; peak GPU memory stays under 1.5 GB


def _tidy(q: np.ndarray) -> np.ndarray:
    """Demand can't be negative, and deciles must not cross."""
    return np.sort(np.maximum(q, 0), axis=1).astype(np.float32)


def seasonal_naive(series: Series, origin: int, horizon: int = HORIZON) -> np.ndarray:
    """Same weekday last week; a point forecast, so only the median is filled."""
    if origin < 7:
        raise ValueError("seasonal naive needs at least 7 days of history")
    q =np.full((len(series.demand), len(DECILES), horizon), np.nan, dtype=np.float32)
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


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.cache
def _chronos2():
    from chronos import Chronos2Pipeline

    repo, revision = CHRONOS2
    return Chronos2Pipeline.from_pretrained(repo, revision=revision, device_map=_device())


@functools.cache
def _timesfm25():
    import timesfm

    repo, revision = TIMESFM25
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=revision)
    model.compile(timesfm.ForecastConfig(
        max_context=128, max_horizon=128, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
        fix_quantile_crossing=True, per_core_batch_size=BATCH,
    ))
    return model


@functools.cache
def _toto2():
    from toto2 import Toto2Model

    repo, revision = TOTO2
    return Toto2Model.from_pretrained(repo, revision=revision).to(_device()).eval()


def release() -> None:
    """Free the loaded foundation models: one at a time fits a 4 GB GPU."""
    for loader in (_chronos2, _timesfm25, _toto2):
        loader.cache_clear()
    if "torch" in sys.modules:
        sys.modules["torch"].cuda.empty_cache()


def chronos2(
    series: Series, origin: int, horizon: int = HORIZON, use_inputs: bool = True
) -> np.ndarray:
    """Chronos-2 zero-shot; the known inputs go in as past and future covariates."""

    def task(i: int) -> dict:
        if not use_inputs:
            return {"target": series.demand[i, :origin]}
        inputs = series.inputs[i, : origin + horizon]
        return {
            "target": series.demand[i, :origin],
            "past_covariates": {c: inputs[:origin, j] for j, c in enumerate(INPUTS)},
            "future_covariates": {c: inputs[origin:, j] for j, c in enumerate(INPUTS)},
        }

    q, _ = _chronos2().predict_quantiles(
        [task(i) for i in range(len(series.demand))], prediction_length=horizon,
        quantile_levels=DECILES.tolist(), batch_size=BATCH,
    )  # one (1, horizon, 9) tensor per series
    return _tidy(np.stack([t[0].T.cpu().numpy() for t in q]))


def timesfm25(series: Series, origin: int, horizon: int = HORIZON) -> np.ndarray:
    """TimesFM-2.5 zero-shot, one series at a time."""
    _, q = _timesfm25().forecast(horizon=horizon, inputs=list(series.demand[:, :origin]))
    return _tidy(q[..., 1:].transpose(0, 2, 1))  # output 0 is the mean, 1-9 the deciles


def toto2(series: Series, origin: int, horizon: int = HORIZON) -> np.ndarray:
    """Toto-2.0 zero-shot, one series at a time; history is left-padded to whole patches."""
    import torch
    from torch.nn.functional import pad

    model = _toto2()
    device = next(model.parameters()).device
    width = -origin % model.config.patch_size
    out = []
    with torch.inference_mode():
        for start in range(0, len(series.demand), BATCH):
            history = np.ascontiguousarray(series.demand[start : start + BATCH, None, :origin])
            target = torch.from_numpy(history).to(device)
            q = model.forecast(
                {
                    "target": pad(target, (width, 0)),
                    "target_mask": pad(torch.ones_like(target, dtype=torch.bool), (width, 0)),
                    "series_ids": torch.zeros(target.shape[:2], dtype=torch.long, device=device),
                },
                horizon=horizon, has_missing_values=True, scaler_fallback_min_obs=8,
            )  # (9, batch, 1, horizon)
            out.append(q[:, :, 0].float().cpu().numpy().transpose(1, 0, 2))
    return _tidy(np.concatenate(out))
