"""Operating-hour arrays (hours 6-21) with stockout hours left blank (NaN)."""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from fresh_replenish.data.db import check_grid

LIST_INDEX = range(7, 23)  # DuckDB lists are 1-based: elements 7..22 are hours 6..21
COVARIATES = ["discount", "holiday_flag", "precpt", "avg_temperature"]

SQL = f"""
SELECT store_id, product_id, dt, sale_amount, stockout_hours, {", ".join(COVARIATES)},
       {", ".join(f"hours_sale[{i}] AS s{i}" for i in LIST_INDEX)},
       {", ".join(f"hours_stock_status[{i}] AS o{i}" for i in LIST_INDEX)}
FROM daily
WHERE split = 'train' {{store_filter}}
ORDER BY store_id, product_id, dt
"""


@dataclass
class Hours:
    ids: pd.DataFrame  # store_id, product_id: one row per series
    dates: np.ndarray  # (days,)
    sales: np.ndarray  # (series, days, 16), NaN where the hour was out of stock
    night: np.ndarray  # (series, days): sales outside 6:00-22:00, always observed
    covariates: np.ndarray  # (series, days, 4) in COVARIATES order
    sale_amount: np.ndarray  # (series, days)
    stockout_hours: np.ndarray  # (series, days)


def load_hours(con: duckdb.DuckDBPyConnection, stores: list[int] | None = None) -> Hours:
    """Train-split hours as arrays; every series must cover the same days."""
    store_filter = "" if stores is None else "AND list_contains($stores::BIGINT[], store_id)"
    params = {} if stores is None else {"stores": list(stores)}
    cols = con.execute(SQL.format(store_filter=store_filter), params).fetchnumpy()
    days, dates = check_grid(cols)
    keys = pd.DataFrame({"store_id": cols["store_id"], "product_id": cols["product_id"]})
    n = len(keys) // days

    raw = np.stack([cols[f"s{i}"] for i in LIST_INDEX], axis=-1).astype(np.float32)
    out = np.stack([cols[f"o{i}"] for i in LIST_INDEX], axis=-1).astype(bool)
    sale_amount = np.asarray(cols["sale_amount"], dtype=np.float32)
    return Hours(
        ids=keys.iloc[::days].reset_index(drop=True),
        dates=dates,
        sales=np.where(out, np.nan, raw).reshape(n, days, 16),
        night=(sale_amount - raw.sum(axis=1)).reshape(n, days),
        covariates=np.stack([cols[c] for c in COVARIATES], axis=-1)
        .astype(np.float32)
        .reshape(n, days, len(COVARIATES)),
        sale_amount=sale_amount.reshape(n, days),
        stockout_hours=np.asarray(cols["stockout_hours"]).reshape(n, days),
    )
