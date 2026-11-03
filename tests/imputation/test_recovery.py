from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest

from fresh_replenish.data.db import connect
from fresh_replenish.imputation import deep, recovery
from fresh_replenish.imputation.masking import mimic_missing


def test_score_and_bootstrap_point_on_hand_example():
    pred = np.array([[11.0, 9.0], [5.0, 0.0]])
    actual = np.array([[10.0, 10.0], [4.0, 4.0]])
    valid = np.array([[True, True], [True, False]])
    point, _ = recovery.bootstrap(recovery.score(pred, actual, valid), n=10)
    assert point["wape"] == pytest.approx(3 / 24)  # |+1| + |-1| + |+1| over 10 + 10 + 4
    assert point["wpe"] == pytest.approx(1 / 24)


def test_bootstrap_interval_contains_point_and_repeats():
    rng = np.random.default_rng(0)
    actual = rng.uniform(5, 10, 50)
    sums = (rng.uniform(0, 2, 50), rng.normal(0, 1, 50), actual)
    point, draws = recovery.bootstrap(sums)
    assert point["wape_low"] <= point["wape"] <= point["wape_high"]
    assert point["wpe_low"] <= point["wpe"] <= point["wpe_high"]
    assert np.array_equal(draws, recovery.bootstrap(sums)[1])


def test_no_recovery_understates_demand():
    rng = np.random.default_rng(0)
    sales = rng.uniform(0.5, 1.5, (20, 30, 16))
    masked, valid = mimic_missing(sales, rng)
    days = valid.any(axis=-1)
    point, _ = recovery.bootstrap(recovery.score(np.nansum(masked, -1), sales.sum(-1), days))
    assert point["wpe"] < -0.1


def _fake_deep(monkeypatch):
    # Deep models are faked here (their own tests train them); autospec keeps the real
    # signatures, so a changed call from run() fails here instead of in the GPU run.
    fake_fit = create_autospec(deep.fit_deep, side_effect=lambda sales, cov, model, **kw: model)
    fake_predict = create_autospec(
        deep.predict_deep, side_effect=lambda imputer, sales, cov, **kw: np.nan_to_num(sales)
    )
    monkeypatch.setattr(recovery, "fit_deep", fake_fit)
    monkeypatch.setattr(recovery, "predict_deep", fake_predict)


def test_run_wires_methods_and_recovers_every_day(tiny_dataset, monkeypatch):
    _fake_deep(monkeypatch)
    results, recovered = recovery.run(connect(*tiny_dataset), sample_share=1.0, epochs=1)
    assert list(results["methods"].method) == [
        "no recovery", "profile", "tobit", "saits", "timesnet"
    ]
    assert results["choice"].chosen.iloc[0] in {"profile", "tobit", "saits", "timesnet"}
    assert len(recovered) == 12
    in_stock = recovered.stockout_hours == 0
    assert (recovered.recovered_demand[in_stock] == recovered.sale_amount[in_stock]).all()


def test_recovered_demand_never_below_recorded_sales(make_dataset, monkeypatch):
    def sold_during_stockout(train, eval_):
        train[0]["hours_sale"][20] = 5.0  # sales recorded in an hour flagged out of stock
        train[0]["sale_amount"] += 5.0

    _fake_deep(monkeypatch)
    _, recovered = recovery.run(
        connect(*make_dataset(sold_during_stockout)), sample_share=1.0, epochs=1
    )
    assert (recovered.recovered_demand >= recovered.sale_amount).all()


def test_report_formats_counts_and_wape():
    cols = ["wape", "wape_low", "wape_high", "wpe", "wpe_low", "wpe_high"]
    methods = pd.DataFrame(
        [["no recovery", 0.709, 0.705, 0.711, -0.709, -0.711, -0.705],
         ["profile", 0.379, 0.372, 0.386, -0.0002, -0.004, 0.004]],
        columns=["method", *cols],
    )
    sample = {"stores": 90, "series": 4928, "valid_days": 76684, "hidden_hour_share": 0.1265}
    report = recovery.render_report({
        "methods": methods,
        "choice": pd.DataFrame([{"chosen": "profile", "reason": "lowest WAPE"}]),
        "sample": pd.DataFrame([{**sample, "epochs": 10}]),
        "timing": pd.DataFrame([("profile", 0.2)], columns=["step", "seconds"]),
    })
    assert "90 stores (4,928 store-product series)" in report
    assert "WAPE 37.9% (95% CI 37.2% to 38.6%)" in report


def test_run_stops_when_nothing_can_be_scored(tiny_dataset, monkeypatch):
    monkeypatch.setattr(recovery, "mimic_missing", lambda x, rng: (x, np.zeros(x.shape, bool)))
    with pytest.raises(ValueError, match="no hidden in-stock day"):
        recovery.run(connect(*tiny_dataset), sample_share=1.0, epochs=1)


def test_main_without_data_returns_2(tmp_path):
    assert recovery.main(raw_dir=tmp_path / "empty", out_dir=tmp_path / "out") == 2
