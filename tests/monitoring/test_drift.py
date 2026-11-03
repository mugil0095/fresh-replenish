import shutil

import numpy as np
import pandas as pd
import pytest

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
    table = drift.monitor(_week(), _before())
    assert list(table.category) == ["all", "0", "1"]
    assert not _alerts(table).to_numpy().any()
    assert table.coverage.tolist() == pytest.approx([0.8] * 3)


@pytest.mark.parametrize(("week", "alert"), [
    (_week(bias=-0.3), "bias_alert"),
    (_week(inside=0.5), "coverage_alert"),
    (_week(stockout=0.4), "stockout_alert"),
])
def test_each_problem_raises_its_alert(week, alert):
    flagged = _alerts(drift.monitor(week, _before())).loc["all"]
    assert flagged[alert] and flagged.sum() == 1


def test_main_without_data_returns_2(tmp_path):
    assert drift.main(raw_dir=tmp_path / "none", processed_dir=tmp_path, out_dir=tmp_path) == 2


def test_main_without_orders_names_the_batch(tiny_dataset, tmp_path, capsys):
    raw = tmp_path / "raw" / "data"
    raw.mkdir(parents=True)
    for path in tiny_dataset:
        shutil.copy(path, raw / path.name)
    assert drift.main(raw_dir=raw.parent, processed_dir=tmp_path, out_dir=tmp_path) == 2
    assert "fresh_replenish.serving.batch" in capsys.readouterr().err
