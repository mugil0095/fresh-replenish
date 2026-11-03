"""DuckDB views over the FreshRetailNet-50K Parquet files (nothing is loaded into RAM)."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# DuckDB lists are 1-based: elements 7..22 are hours 6..21, the stores' operating window.
DAILY_VIEW = """
CREATE OR REPLACE VIEW daily AS
SELECT *,
       len(hours_sale) AS n_hours,
       len(hours_stock_status) AS n_status,
       list_sum(list_slice(hours_stock_status, 7, 22)) AS stockout_hours,
       list_sum(hours_sale) AS hours_sale_sum
FROM (
    SELECT 'train' AS split, * FROM train
    UNION ALL
    SELECT 'eval' AS split, * FROM eval
)
"""


def connect(train_path: Path, eval_path: Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with views `train`, `eval` and `daily` over the two Parquet files."""
    con = duckdb.connect()
    # read_parquet takes the path as a value, so folders with spaces need no SQL quoting.
    con.read_parquet(str(train_path)).create_view("train")
    con.read_parquet(str(eval_path)).create_view("eval")
    con.execute(DAILY_VIEW).fetchall()  # consume the result: DuckDB 2 blocks half-read connections
    return con


def sample_stores(
    con: duckdb.DuckDBPyConnection, share: float, rng: np.random.Generator
) -> list[int]:
    """A random share of the train stores (at least one), sorted."""
    stores = np.sort(con.execute("SELECT DISTINCT store_id FROM train").fetchnumpy()["store_id"])
    size = max(1, round(len(stores) * share))
    return sorted(rng.choice(stores, size=size, replace=False).tolist())


def check_grid(cols: dict) -> tuple[int, np.ndarray]:
    """Days per series and their shared dates, for rows sorted by store, product and date."""
    keys = pd.DataFrame({"store_id": cols["store_id"], "product_id": cols["product_id"]})
    counts = keys.value_counts(sort=False)
    if counts.nunique() != 1:
        raise ValueError("every series must have the same number of train days")
    days = int(counts.iloc[0])
    dates = np.asarray(cols["dt"]).reshape(-1, days)
    if not (dates == dates[0]).all():
        raise ValueError("series must cover the same dates")
    return days, dates[0]
