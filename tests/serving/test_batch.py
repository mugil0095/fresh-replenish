import shutil

import numpy as np
import pandas as pd
import pytest

from fresh_replenish.data.db import connect
from fresh_replenish.metrics import DECILE_COLUMNS, REST_COLUMNS
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


def _plan_shifts(group, shift):
    return _shifts(group, shift).assign(decision_hour=13, arrival_hour=15, morning_share=0.8)


def _left(series, share=0.5):
    return series.ids[["store_id", "product_id"]].assign(left=share)


def test_targets_scale_the_day_to_what_is_left(make_series, weekly):
    s = make_series(weekly(n=4, days=14))
    orders = batch.forecast_orders(s, 7, _shifts(0, 0.0), _fixed, ratio=9)
    served = batch.with_top_up(orders, _left(s), s.ids, _plan_shifts(0, 0.0))
    assert served[REST_COLUMNS].iloc[0].tolist() == pytest.approx(np.linspace(0.5, 4.5, 9))
    assert (served.decision_hour == 13).all() and (served.arrival_hour == 15).all()
    assert (served.morning_share == 0.8).all()
    assert served[orders.columns].equals(orders)


def test_targets_use_the_pooled_shift_for_an_unseen_category(make_series, weekly):
    s = make_series(weekly(n=4, days=14))
    orders = batch.forecast_orders(s, 7, _shifts(0, 0.0), _fixed, ratio=9)
    served = batch.with_top_up(orders, _left(s), s.ids, _plan_shifts(POOLED, 0.5))
    assert (served.rest_q90 == 6.5).all()  # 4.5 + 0.5 x spread of 4


def test_every_order_needs_a_usual_share(make_series, weekly):
    s = make_series(weekly(n=4, days=14))
    orders = batch.forecast_orders(s, 7, _shifts(0, 0.0), _fixed, ratio=9)
    with pytest.raises(ValueError, match="share"):
        batch.with_top_up(orders, _left(s).iloc[1:], s.ids, _plan_shifts(0, 0.0))


def test_runs_are_tracked_in_mlflow(tmp_path):
    import mlflow

    batch.log_run(tmp_path / "mlruns", "batch", {"ratio": 9}, {"runtime_s": 1.5})
    runs = mlflow.search_runs(experiment_names=["fresh-replenish"])
    assert runs["params.ratio"].tolist() == ["9"] and runs["metrics.runtime_s"].tolist() == [1.5]


def test_main_names_the_missing_top_up_plan(tiny_dataset, tmp_path, capsys):
    raw = tmp_path / "raw" / "data"
    raw.mkdir(parents=True)
    for path in tiny_dataset:
        shutil.copy(path, raw / path.name)
    (tmp_path / "tables").mkdir()
    (tmp_path / "recovered_demand.parquet").touch()
    (tmp_path / "tables" / "calibration_shifts.csv").touch()
    code = batch.main(raw_dir=raw.parent, processed_dir=tmp_path, out_dir=tmp_path)
    assert code == 2 and "decisions.topup" in capsys.readouterr().err


def test_main_without_data_returns_2(tmp_path):
    assert batch.main(raw_dir=tmp_path / "none", processed_dir=tmp_path, out_dir=tmp_path) == 2
