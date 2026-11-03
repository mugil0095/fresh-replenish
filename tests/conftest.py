"""Shared fixtures: tiny FreshRetailNet-50K files with the real schema, and synthetic series."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fresh_replenish.data.db import connect
from fresh_replenish.data.download import local_paths
from fresh_replenish.forecasting.data import ATTRIBUTES, Series, load_series

SCHEMA = pa.schema(
    [
        ("city_id", pa.int64()),
        ("store_id", pa.int64()),
        ("management_group_id", pa.int64()),
        ("first_category_id", pa.int64()),
        ("second_category_id", pa.int64()),
        ("third_category_id", pa.int64()),
        ("product_id", pa.int64()),
        ("dt", pa.string()),
        ("sale_amount", pa.float64()),
        ("hours_sale", pa.list_(pa.float64())),
        ("stock_hour6_22_cnt", pa.int32()),
        ("hours_stock_status", pa.list_(pa.int64())),
        ("discount", pa.float64()),
        ("holiday_flag", pa.int32()),
        ("activity_flag", pa.int32()),
        ("precpt", pa.float64()),
        ("avg_temperature", pa.float64()),
        ("avg_humidity", pa.float64()),
        ("avg_wind_level", pa.float64()),
    ]
)


def _row(store: int, product: int, dt: str) -> dict:
    hours_sale = [0.0] * 24
    hours_sale[12] = 0.5
    return {
        "city_id": store - 1,
        "store_id": store,
        "management_group_id": 0,
        "first_category_id": 5 if product == 10 else 6,
        "second_category_id": 1,
        "third_category_id": 1,
        "product_id": product,
        "dt": dt,
        "sale_amount": 0.5,
        "hours_sale": hours_sale,
        "stock_hour6_22_cnt": 0,
        "hours_stock_status": [0] * 24,
        "discount": 1.0,
        "holiday_flag": 0,
        "activity_flag": 0,
        "precpt": 0.0,
        "avg_temperature": 25.0,
        "avg_humidity": 80.0,
        "avg_wind_level": 1.5,
    }


def base_rows() -> tuple[list[dict], list[dict]]:
    """12 train rows (2 stores x 2 products x 3 days) and 4 eval rows (1 day)."""
    pairs = [(s, p) for s in (1, 2) for p in (10, 11)]
    train = [_row(s, p, f"2024-01-0{d}") for s, p in pairs for d in (1, 2, 3)]
    eval_ = [_row(s, p, "2024-01-04") for s, p in pairs]
    # Mirrors real eval row 0: closed at night (flag 1), 2 stockout hours inside 6-21.
    first = train[0]
    first["hours_stock_status"] = [1, 1, 1] + [0] * 17 + [1, 1, 1, 1]
    first["stock_hour6_22_cnt"] = 2
    first["hours_sale"] = [0.0] * 24
    first["hours_sale"][10] = first["hours_sale"][19] = 0.1
    first["sale_amount"] = 0.2
    return train, eval_


@pytest.fixture
def make_dataset(tmp_path):
    """Return make(mutate=None) -> (train_path, eval_path); mutate(train, eval) edits rows first."""

    def make(mutate=None):
        train, eval_ = base_rows()
        if mutate:
            mutate(train, eval_)
        folder = tmp_path / "data dir with spaces"
        folder.mkdir(exist_ok=True)
        paths = folder / "train.parquet", folder / "eval.parquet"
        for rows, path in zip((train, eval_), paths, strict=True):
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        return paths

    return make


@pytest.fixture
def tiny_dataset(make_dataset):
    return make_dataset()


@pytest.fixture
def real_paths():
    """The downloaded dataset files; skips the test when they aren't downloaded."""
    paths = local_paths()
    if paths is None:
        pytest.skip("run `uv run python -m fresh_replenish.data.download` first")
    return paths

@pytest.fixture
def weekly():
    """weekly(n, days, seed): positive demand with a weekly pattern, a level per series, noise."""
    return _weekly


def _weekly(n: int = 20, days: int = 63, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = rng.uniform(2, 10, (n, 1))
    pattern = 1 + 0.5 * np.sin(2 * np.pi * np.arange(days) / 7)
    return level * pattern * rng.gamma(20, 1 / 20, (n, days))


@pytest.fixture
def make_series():
    """make_series(demand, inputs=None) -> Series with synthetic ids and dates."""

    def make(demand, inputs=None):
        n, days = demand.shape
        ids = pd.DataFrame({
            "store_id": np.arange(n) // 2,
            "product_id": np.arange(n) % 2 + 10,
            **{c: np.zeros(n, dtype=int) for c in ATTRIBUTES},
        })
        return Series(
            ids=ids,
            dates=np.arange(days).astype(str),
            demand=demand.astype(np.float32),
            in_stock=np.ones((n, days), dtype=bool),
            inputs=np.zeros((n, days, 3), np.float32) if inputs is None else inputs,
        )

    return make


@pytest.fixture
def no_peeking():
    """no_peeking(forecaster, series, origin): demand changed from the origin on changes nothing."""
    return _no_peeking


def _no_peeking(forecaster, series, origin, horizon=7):
    before = forecaster(series, origin, horizon)
    series.demand[:, origin:] *= 3
    # A tolerance, not equality: GPU kernels may differ in the last bits between calls.
    np.testing.assert_allclose(forecaster(series, origin, horizon), before, rtol=1e-4, atol=1e-6)


@pytest.fixture
def series_from(tmp_path):
    """series_from(paths, **kw): writes a recovered-demand file like Phase 3's, then loads it."""

    def write(split_path, name):
        rows = pd.read_parquet(split_path)
        recovered = rows[["store_id", "product_id", "dt", "sale_amount"]].assign(
            stockout_hours=rows.stock_hour6_22_cnt,
            recovered_demand=rows.sale_amount + (rows.stock_hour6_22_cnt > 0),
        )
        recovered.to_parquet(tmp_path / name, index=False)
        return tmp_path / name

    def load(paths, with_eval=False, **kwargs):
        files = [write(paths[0], "recovered.parquet")]
        if with_eval:
            files.append(write(paths[1], "recovered_eval.parquet"))
        return load_series(connect(*paths), files, **kwargs)

    return load
