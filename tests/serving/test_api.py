import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fresh_replenish.metrics import DECILE_COLUMNS
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


def test_health_reports_loaded_orders(orders_path, tmp_path):
    assert TestClient(create_app(orders_path)).get("/health").json() == {
        "status": "ok", "orders_loaded": True, "as_of": "2024-06-25"
    }
    assert TestClient(create_app(tmp_path / "none.parquet")).get("/health").json()[
        "orders_loaded"
    ] is False


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


def test_no_orders_file_means_service_unavailable(tmp_path):
    client = TestClient(create_app(tmp_path / "none.parquet"))
    assert client.get("/reorder", params={"store_id": 1}).status_code == 503


def test_latency_is_measured_over_http(orders_path):
    result = latency.measure(create_app(orders_path), [1, 2], requests=20)
    assert result["requests"] == 20 and 0 < result["p50_ms"] <= result["p95_ms"]
