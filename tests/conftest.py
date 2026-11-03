"""Tiny synthetic FreshRetailNet-50K files with the real 19-column schema."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fresh_replenish.data.download import local_paths

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
