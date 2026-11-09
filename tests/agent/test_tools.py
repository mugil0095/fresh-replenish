import json

import numpy as np
import pytest

from fresh_replenish.agent.tools import Toolbox, chronos_scenario
from fresh_replenish.config import PROCESSED_DIR
from fresh_replenish.data.db import connect


def test_a_shelf_count_is_topped_up_to_the_rest_of_day_target(toolbox):
    out = toolbox.compute_top_up(1, 10, "2024-06-26", shelf=1.5)
    assert out == {
        "store_id": 1, "product_id": 10, "date": "2024-06-26", "ratio": 9.0,
        "critical_ratio": 0.9, "shelf": 1.5, "target": 4.5, "top_up": 3.0,
        "decision_hour": 13, "arrival_hour": 15,
    }
    json.dumps(out)  # the model reads it as JSON


def test_a_full_shelf_needs_no_top_up(toolbox):
    assert toolbox.compute_top_up(1, 10, "2024-06-26", shelf=6.0)["top_up"] == 0


def test_the_top_up_target_follows_the_cost_ratio(toolbox):
    assert toolbox.compute_top_up(1, 10, "2024-06-26", shelf=0, ratio=1)["target"] == 2.5


@pytest.mark.parametrize(
    ("product", "shelf", "ratio"),
    [(99, 1.0, 9.0), (10, -1.0, 9.0), (10, float("nan"), 9.0), (10, 1.0, 20.0)],
)
def test_a_top_up_refuses_what_it_cannot_answer(toolbox, product, shelf, ratio):
    assert "error" in toolbox.compute_top_up(1, product, "2024-06-26", shelf, ratio)


def test_forecast_lookup(toolbox):
    assert toolbox.get_forecast(1, 10, "2024-06-26") == {
        "store_id": 1, "product_id": 10, "date": "2024-06-26", "as_of": "2024-06-25",
        "low": 1.0, "median": 5.0, "high": 9.0,
    }


def test_missing_data_is_an_error_not_a_guess(toolbox):
    assert "no forecast" in toolbox.get_forecast(1, 99, "2024-06-26")["error"]
    assert "no forecast" in toolbox.compute_order(1, 10, "2025-01-01")["error"]


def test_cost_based_order_at_the_ratio(toolbox):
    order = toolbox.compute_order(1, 10, "2024-06-26")
    assert (order["order"], order["critical_ratio"], order["within_limits"]) == (9.0, 0.9, True)
    assert toolbox.compute_order(1, 10, "2024-06-26", ratio=1)["order"] == 5.0


def test_a_quantity_over_the_line_limit_fails_the_check(toolbox):
    order = toolbox.compute_order(1, 10, "2024-06-26", units=20)
    assert order["line_limit"] == 13.5 and order["within_limits"] is False


def test_store_capacity_counts_the_other_lines(orders, tmp_path):
    toolbox = Toolbox(orders, records=tmp_path / "records.jsonl", capacity=15)
    order = toolbox.compute_order(1, 10, "2024-06-26")
    assert order["store_day_total"] == 18.0 and order["within_limits"] is False


def test_saved_orders_count_toward_store_capacity(orders, tmp_path):
    records = tmp_path / "records.jsonl"
    toolbox = Toolbox(orders, records=records, capacity=20)  # store 1 plans 9 + 9 on 06-26
    toolbox.write_decision_record(1, 10, "2024-06-26", 10.0, "more", [])
    assert toolbox.compute_order(1, 11, "2024-06-26", units=11)["store_day_total"] == 21.0
    later = Toolbox(orders, records=records, capacity=20)  # a new session reads the saved order
    assert later.compute_order(1, 11, "2024-06-26", units=11)["within_limits"] is False


def test_a_negative_quantity_is_refused(toolbox):
    assert "units" in toolbox.compute_order(1, 10, "2024-06-26", units=-40).get("error", "")


def test_bad_ratio_is_refused(toolbox):
    assert "ratio" in toolbox.compute_order(1, 10, "2024-06-26", ratio=20)["error"]


def test_what_if_is_labelled_a_model_estimate(toolbox):
    result = toolbox.what_if(1, 10, "2024-06-26", discount=0.8)
    assert result["change_pct"] == 25.0 and "model-based estimate" in result["note"]
    assert "error" in toolbox.what_if(1, 10, "2024-06-26")  # no change asked for
    assert "error" in toolbox.what_if(1, 99, "2024-06-26", promotion=1)


def test_decision_record_keeps_order_reason_and_tool_outputs(toolbox):
    toolbox.write_decision_record(1, 10, "2024-06-26", 9.0, "covers 90% of demand", [{"a": 1}])
    record = json.loads(toolbox.records.read_text(encoding="utf-8").splitlines()[0])
    assert (record["order"], record["reason"], record["tool_outputs"]) == (
        9.0, "covers 90% of demand", [{"a": 1}]
    )


@pytest.mark.data
def test_chronos_scenario_changes_only_the_asked_input(real_paths):
    recovered = PROCESSED_DIR / "recovered_demand.parquet"
    if not recovered.exists():
        pytest.skip("run `uv run python -m fresh_replenish.imputation.recovery` first")

    def promo_doubles(series, origin, horizon):  # median = 1 + the promotion flag
        q = np.ones((len(series.ids), 9, horizon), np.float32)
        return q + series.inputs[:, origin : origin + horizon, 1][:, None, :]

    con = connect(*real_paths)
    product = con.execute("SELECT min(product_id) FROM train WHERE store_id = 0").fetchall()[0][0]
    run = chronos_scenario(con, recovered, forecaster=promo_doubles)
    base, changed = run(0, product, "2024-06-27", {"activity_flag": 1})
    assert changed == 2.0 and base in (1.0, 2.0)
    assert run(0, product, "2024-07-30", {"activity_flag": 1}) is None  # outside the week
