import numpy as np
import pandas as pd
import pytest

from fresh_replenish.data.db import connect
from fresh_replenish.metrics import DECILE_COLUMNS
from fresh_replenish.serving import batch
from fresh_replenish.uncertainty.conformal import POOLED


def test_future_inputs_follow_the_series(make_dataset, series_from):
    paths = make_dataset(lambda train, eval_: eval_[0].update(discount=0.7, activity_flag=1))
    future = batch.with_future(connect(*paths), series_from(paths))
    assert future.demand.shape == (4, 4) and np.isnan(future.demand[:, 3]).all()
    assert future.inputs[0, 3].tolist() == pytest.approx([0.7, 1.0, 0.0])
    assert future.dates[-1] == "2024-01-04"


def _fixed(series, origin, horizon):
    return np.broadcast_to(np.linspace(1, 9, 9)[None, :, None], (len(series.ids), 9, horizon))


def _shifts(group, shift):
    row = {"model": "lightgbm", "group": group, **dict.fromkeys(DECILE_COLUMNS, shift)}
    return pd.DataFrame([row])


def test_orders_sit_at_the_critical_ratio(make_series, weekly):
    s = make_series(weekly(n=4, days=14))  # category 0
    orders = batch.forecast_orders(s, 7, _shifts(0, 0.0), _fixed, ratio=9)
    assert len(orders) == 4 * 7 and (orders.order == 9).all()  # q90
    assert (batch.forecast_orders(s, 7, _shifts(0, 0.0), _fixed, ratio=1).order == 5).all()


def test_unseen_category_uses_the_pooled_shift(make_series, weekly):
    s = make_series(weekly(n=4, days=14))
    orders = batch.forecast_orders(s, 7, _shifts(POOLED, 0.5), _fixed, ratio=9)
    assert (orders.order == 13).all()  # q90 + 0.5 x spread of 8


def test_runs_are_tracked_in_mlflow(tmp_path):
    import mlflow

    batch.log_run(tmp_path / "mlruns", "batch", {"ratio": 9}, {"runtime_s": 1.5})
    runs = mlflow.search_runs(experiment_names=["fresh-replenish"])
    assert runs["params.ratio"].tolist() == ["9"] and runs["metrics.runtime_s"].tolist() == [1.5]


def test_main_without_data_returns_2(tmp_path):
    assert batch.main(raw_dir=tmp_path / "none", processed_dir=tmp_path, out_dir=tmp_path) == 2
