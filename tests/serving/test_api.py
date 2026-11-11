import os
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fresh_replenish.metrics import DECILE_COLUMNS, REST_COLUMNS
from fresh_replenish.serving import latency
from fresh_replenish.serving.api import create_app


@pytest.fixture
def orders_path(tmp_path):
    """Orders for 2 stores x 2 products x 2 days; deciles 1 ... 9, ordered at q90."""
    rows = pd.DataFrame(
        [(s, p, f"2024-06-2{6 + d}") for s in (1, 2) for p in (10, 11) for d in (0, 1)],
        columns=["store_id", "product_id", "dt"],
    )
    rows[DECILE_COLUMNS] = np.linspace(1, 9, 9)
    path = tmp_path / "orders.parquet"
    rows.assign(order=9.0, as_of="2024-06-25", ratio=9.0).to_parquet(path, index=False)
    return path


@pytest.fixture
def client(orders_path):
    """Orders with rest-of-day targets 0.5 ... 4.5, as the batch writes them."""
    rows = pd.read_parquet(orders_path)
    rows[REST_COLUMNS] = np.linspace(0.5, 4.5, 9)
    rows.assign(decision_hour=13, arrival_hour=15, morning_share=0.8).to_parquet(orders_path)
    return TestClient(create_app(orders_path))


COUNT = {"store_id": 1, "date": "2024-06-26", "shelf": {"10": 1.5, "11": 6.0}}


def test_a_shelf_count_is_topped_up_to_the_target(client):
    body = client.post("/topup", json=COUNT).json()
    assert body["critical_ratio"] == pytest.approx(0.9) and body["as_of"] == "2024-06-25"
    assert (body["decision_hour"], body["arrival_hour"]) == (13, 15)
    assert body["lines"] == [
        {"product_id": 10, "shelf": 1.5, "on_order": 0.0, "target": 4.5, "top_up": 3.0},
        {"product_id": 11, "shelf": 6.0, "on_order": 0.0, "target": 4.5, "top_up": 0.0},
    ]


def test_a_second_count_does_not_order_the_same_units_again(client):
    again = COUNT | {"on_order": {"10": 3.0}}  # the first count's top-up hasn't landed yet
    assert client.post("/topup", json=again).json()["lines"][0]["top_up"] == 0.0
    part = client.post("/topup", json=COUNT | {"on_order": {"10": 1.0}}).json()["lines"][0]
    assert (part["on_order"], part["top_up"]) == (1.0, 2.0)  # the reply shows its sums


def test_a_count_at_the_plans_hour_is_accepted(client):
    assert client.post("/topup", json=COUNT | {"hour": 13}).status_code == 200


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"hour": 16}, "counted at 13:00"),
        ({"on_order": {"99": 1.0}}, "on order but not counted: [99]"),
        ({"on_order": {"10": -1.0}}, "greater than or equal to 0"),
    ],
)
def test_counts_the_targets_cannot_answer_are_refused(client, change, message):
    response = client.post("/topup", json=COUNT | change)
    assert response.status_code == 422 and message in response.text


def test_a_lower_ratio_tops_up_less(client):
    body = client.post("/topup", json=COUNT | {"ratio": 1}).json()
    assert [line["target"] for line in body["lines"]] == [2.5, 2.5]  # the median


def test_only_the_counted_lines_come_back(client):
    body = client.post("/topup", json=COUNT | {"shelf": {"11": 0}}).json()
    assert [line["product_id"] for line in body["lines"]] == [11]


@pytest.mark.parametrize(("change", "status", "message"), [
    ({"store_id": 99}, 404, "no orders for store 99"),
    ({"date": "2024-07-09"}, 404, "on 2024-07-09"),
    ({"shelf": {"10": 1, "77": 2, "78": 1}}, 422, "[77, 78]"),
    ({"shelf": {"10": -1}}, 422, "greater than or equal to 0"),
    ({"shelf": {}}, 422, "at least 1 item"),
    ({"ratio": 20}, 422, "less than or equal to 9"),
])
def test_bad_shelf_counts(client, change, status, message):
    response = client.post("/topup", json=COUNT | change)
    assert response.status_code == status and message in response.text


def test_a_count_that_is_not_a_number_is_refused(client):
    body = '{"store_id": 1, "date": "2024-06-26", "shelf": {"10": NaN}}'
    response = client.post("/topup", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_orders_without_targets_ask_for_a_new_batch(orders_path):
    response = TestClient(create_app(orders_path)).post("/topup", json=COUNT)
    assert response.status_code == 503 and "run the batch job" in response.text


def test_no_orders_file_means_no_top_up(tmp_path):
    response = TestClient(create_app(tmp_path / "none.parquet")).post("/topup", json=COUNT)
    assert response.status_code == 503


def test_reorder_says_how_much_to_send_in_the_morning(client, orders_path):
    assert client.get("/reorder", params={"store_id": 1}).json()["morning_share"] == 0.8
    old = pd.read_parquet(orders_path).drop(columns="morning_share")
    old.to_parquet(orders_path)
    os.utime(orders_path, (time.time() + 60,) * 2)
    assert client.get("/reorder", params={"store_id": 1}).json()["morning_share"] is None


def test_health_reports_loaded_orders(orders_path, tmp_path):
    assert TestClient(create_app(orders_path)).get("/health").json() == {
        "status": "ok", "orders_loaded": True, "as_of": "2024-06-25"
    }
    missing = TestClient(create_app(tmp_path / "none.parquet")).get("/health")
    assert missing.status_code == 503 and missing.json()["orders_loaded"] is False


def test_each_new_batch_is_served_without_a_restart(orders_path, tmp_path):
    path = tmp_path / "served.parquet"
    client = TestClient(create_app(path))
    assert client.get("/reorder", params={"store_id": 1}).status_code == 503
    path.write_bytes(orders_path.read_bytes())
    assert client.get("/health").json()["as_of"] == "2024-06-25"
    pd.read_parquet(path).assign(as_of="2024-07-02").to_parquet(path)
    os.utime(path, (time.time() + 60,) * 2)  # a later batch, even on a coarse file clock
    assert client.get("/reorder", params={"store_id": 1}).json()["as_of"] == "2024-07-02"


def test_reorder_returns_the_store_lines(orders_path):
    body = TestClient(create_app(orders_path)).get("/reorder", params={"store_id": 1}).json()
    assert body["critical_ratio"] == pytest.approx(0.9) and len(body["lines"]) == 4
    line = body["lines"][0]
    assert (line["order"], line["low"], line["median"], line["high"]) == (9, 1, 5, 9)


def test_a_lower_ratio_orders_less(orders_path):
    response = TestClient(create_app(orders_path)).get(
        "/reorder", params={"store_id": 1, "ratio": 1}
    )
    assert {line["order"] for line in response.json()["lines"]} == {5.0}  # the median


@pytest.mark.parametrize(("params", "status"), [
    ({"store_id": 99}, 404),
    ({"store_id": 1, "ratio": 20}, 422),
    ({"store_id": 1, "ratio": 0.05}, 422),
])
def test_bad_requests(orders_path, params, status):
    assert TestClient(create_app(orders_path)).get("/reorder", params=params).status_code == status


def test_an_empty_orders_file_counts_as_no_orders(orders_path):
    import pandas as pd

    pd.read_parquet(orders_path).iloc[:0].to_parquet(orders_path)
    assert TestClient(create_app(orders_path)).get("/health").json()["orders_loaded"] is False


def test_no_orders_file_means_service_unavailable(tmp_path):
    client = TestClient(create_app(tmp_path / "none.parquet"))
    assert client.get("/reorder", params={"store_id": 1}).status_code == 503


def test_latency_gives_up_when_the_server_cannot_start(orders_path, monkeypatch):
    monkeypatch.setattr(latency.uvicorn.Server, "run", lambda self: None)  # never starts
    with pytest.raises(RuntimeError, match="did not start"):
        latency.measure(create_app(orders_path), [1], requests=1, start_timeout=0.2)


def test_latency_can_measure_another_call(client):
    calls = []

    def call(http, store):
        calls.append(store)
        return http.post("/topup", json=COUNT)

    result = latency.measure(client.app, [1], requests=3, call=call)
    assert calls == [1, 1, 1] and result["requests"] == 3


def test_latency_is_measured_over_http(orders_path):
    result = latency.measure(create_app(orders_path), [1, 2], requests=20)
    assert result["requests"] == 20 and 0 < result["p50_ms"] <= result["p95_ms"]
