import shutil

import numpy as np
import pandas as pd
import pytest

from fresh_replenish.metrics import DECILE_COLUMNS, REST_COLUMNS
from fresh_replenish.monitoring import drift

ALERTS = ["bias_alert", "coverage_alert", "stockout_alert"]


def _week(bias=0.0, inside=0.8, stockout=0.2, n=100):
    """Forecasts and outcomes for one week; rows alternate between categories 0 and 1."""
    i = np.arange(n)
    actual = np.linspace(1, 3, n)
    return pd.DataFrame({
        "first_category_id": i % 2,
        "actual": actual,
        "q10": np.where(i < inside * n, actual - 1, actual + 0.5),  # the rest miss the range
        "q50": actual * (1 + bias),
        "q90": actual + 1,
        "stockout": (i < stockout * n).astype(int),
    })


def _before(stockout=0.2, n=100):
    i = np.arange(n)
    return pd.DataFrame({"first_category_id": i % 2, "stockout": (i < stockout * n).astype(int)})


def _alerts(table):
    return table.set_index("category")[ALERTS]


def test_a_healthy_week_raises_no_alert():
    table = drift.monitor(_week(bias=-0.12), _before())  # the median's usual low bias
    assert list(table.category) == ["all", "0", "1"]
    assert not _alerts(table).to_numpy().any()
    assert table.coverage.tolist() == pytest.approx([0.8] * 3)


@pytest.mark.parametrize(("week", "alert"), [
    (_week(bias=-0.3), "bias_alert"),
    (_week(bias=0.1), "bias_alert"),
    (_week(inside=0.5), "coverage_alert"),
    (_week(stockout=0.4), "stockout_alert"),
])
def test_each_problem_raises_its_alert(week, alert):
    flagged = _alerts(drift.monitor(week, _before())).loc["all"]
    assert flagged[alert] and flagged.sum() == 1


def test_report_shows_the_served_second_delivery():
    table = drift.monitor(_week(), _before())
    second = pd.DataFrame([{
        "series": 50000, "decision_hour": 13, "arrival_hour": 15, "cost": 0.6,
        "one_delivery_cost": 0.9, "saving": 1 / 3, "saving_low": 0.3, "saving_high": 0.35,
        "saving_no_stockout": 0.2,
    }])
    report = drift.render_report(table, "2024-06-25", second)
    assert "50,000 series" in report and "33% less than one delivery" in report
    assert "second delivery" not in drift.render_report(table, "2024-06-25").lower()


def test_main_replays_the_served_plan(tiny_dataset, tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "data"
    raw.mkdir(parents=True)
    for path in tiny_dataset:
        shutil.copy(path, raw / path.name)
    orders = pd.DataFrame({
        "store_id": [1, 1, 2, 2], "product_id": [10, 11, 10, 11], "dt": "2024-01-04",
        "as_of": "2024-01-03", "decision_hour": 13, "arrival_hour": 15, "morning_share": 0.8,
    })
    orders[DECILE_COLUMNS] = np.linspace(0.1, 0.9, 9)
    orders[REST_COLUMNS] = np.linspace(0.1, 0.9, 9)
    orders.to_parquet(tmp_path / "orders.parquet")
    logged = {}
    monkeypatch.setattr(drift, "log_run", lambda *args: logged.update(args[3]))
    assert drift.main(raw_dir=raw.parent, processed_dir=tmp_path, out_dir=tmp_path) == 0
    saved = pd.read_csv(tmp_path / "tables" / "monitoring_second_delivery.csv").iloc[0]
    assert saved.series == 4 and saved.decision_hour == 13
    assert "Second delivery, as served" in (tmp_path / "monitoring.md").read_text(encoding="utf-8")
    assert logged["second_delivery_saving"] == saved.saving


def test_main_without_data_returns_2(tmp_path):
    assert drift.main(raw_dir=tmp_path / "none", processed_dir=tmp_path, out_dir=tmp_path) == 2


def test_main_without_orders_names_the_batch(tiny_dataset, tmp_path, capsys):
    raw = tmp_path / "raw" / "data"
    raw.mkdir(parents=True)
    for path in tiny_dataset:
        shutil.copy(path, raw / path.name)
    assert drift.main(raw_dir=raw.parent, processed_dir=tmp_path, out_dir=tmp_path) == 2
    assert "fresh_replenish.serving.batch" in capsys.readouterr().err


def test_tiny_categories_do_not_alert():
    week = pd.concat([_week(), _week(bias=-0.5, n=10).assign(first_category_id=9)])
    flagged = _alerts(drift.monitor(week, _before())).loc["9"]
    assert not flagged.any()  # 10 series-days is too few to trust
