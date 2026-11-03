import json

import numpy as np
import pandas as pd
import pytest

from fresh_replenish.dashboard import data
from fresh_replenish.metrics import CALIBRATED_COLUMNS


def _forecasts():
    """Two series in two categories: deciles 1 ... 9 around an actual demand of 5."""
    rows = pd.DataFrame({
        "model": "lightgbm", "store_id": [1, 1], "product_id": [10, 11],
        "dt": ["2024-06-26"] * 2, "actual": [5.0, 12.0],
    })
    rows[CALIBRATED_COLUMNS] = np.linspace(1, 9, 9)
    return rows


def test_losses_split_waste_and_lost_sales_by_category():
    categories = pd.DataFrame({"store_id": [1, 1], "product_id": [10, 11],
                               "first_category_id": [7, 8]})
    out = data.losses(_forecasts(), categories, ratio=9).set_index("first_category_id")
    # order = q90 = 9: product 10 wastes 4 units; product 11 misses 3 sales at 9 each
    assert (out.loc[7, "waste_cost"], out.loc[7, "lost_sales_cost"]) == (4.0, 0.0)
    assert (out.loc[8, "waste_cost"], out.loc[8, "lost_sales_cost"]) == (0.0, 27.0)
    assert list(out.index) == [8, 7]  # costliest first


def test_headline_compares_cost_based_orders_with_the_median_and_last_week():
    policies = pd.DataFrame({
        "ratio": [9.0] * 3, "policy": ["seasonal naive", "lightgbm median", "lightgbm"],
        "cost": [2.0, 2.5, 1.0],
    })
    h = data.headline(policies)
    assert h["model"] == "lightgbm"
    assert (h["vs_median"], h["vs_naive"]) == pytest.approx((0.6, 0.5))


def test_records_newest_first_and_empty_when_missing(tmp_path):
    path = tmp_path / "records.jsonl"
    lines = [{"time": t, "store_id": 1, "product_id": 2, "date": "2024-06-26", "order": 1.5,
              "reason": "r", "tool_outputs": []} for t in ("2026-11-01", "2026-11-02")]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    assert data.records(path).time.tolist() == ["2026-11-02", "2026-11-01"]
    assert data.records(tmp_path / "none.jsonl").empty
