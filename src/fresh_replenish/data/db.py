"""DuckDB views over the FreshRetailNet-50K Parquet files (nothing is loaded into RAM)."""

from pathlib import Path

import duckdb

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
    con.execute(DAILY_VIEW)
    return con
