"""Daily demand series for forecasting: Phase 3's recovered demand plus inputs known in advance."""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fresh_replenish.data.db import check_grid

INPUTS = ["discount", "activity_flag", "holiday_flag"]  # planned ahead; weather is known only after
ATTRIBUTES = [
    "city_id", "management_group_id", "first_category_id", "second_category_id",
    "third_category_id",
]
HORIZON = 7
ORIGINS = (62, 69, 76, 83)  # weekly folds; the last one forecasts the final 7 train days

SQL = f"""
SELECT store_id, product_id, dt, r.recovered_demand, r.stockout_hours,
       {", ".join(f"t.{c}" for c in ATTRIBUTES + INPUTS)}
FROM recovered r
JOIN daily t USING (store_id, product_id, dt)
{{store_filter}}
ORDER BY store_id, product_id, dt
"""


@dataclass
class Series:
    ids: pd.DataFrame  # store_id, product_id and ATTRIBUTES: one row per series
    dates: np.ndarray  # (days,)
    demand: np.ndarray  # (series, days): recovered daily demand
    in_stock: np.ndarray  # (series, days): no operating hour was out of stock
    inputs: np.ndarray  # (series, days, 3) in INPUTS order


def load_series(
    con: duckdb.DuckDBPyConnection, recovered: Path | list[Path], stores: list[int] | None = None
) -> Series:
    """Recovered demand as arrays, for the days in the file(s); every series must cover them all."""
    files = [str(p) for p in recovered] if isinstance(recovered, list) else str(recovered)
    con.read_parquet(files).create_view("recovered")
    store_filter = "" if stores is None else "WHERE list_contains($stores::BIGINT[], store_id)"
    params = {} if stores is None else {"stores": list(stores)}
    cols = con.execute(SQL.format(store_filter=store_filter), params).fetchnumpy()
    days, dates = check_grid(cols)
    n = len(cols["dt"]) // days
    ids = pd.DataFrame({c: cols[c] for c in ["store_id", "product_id", *ATTRIBUTES]})
    return Series(
        ids=ids.iloc[::days].reset_index(drop=True),
        dates=dates,
        demand=np.asarray(cols["recovered_demand"], dtype=np.float32).reshape(n, days),
        in_stock=(np.asarray(cols["stockout_hours"]) == 0).reshape(n, days),
        inputs=np.stack([cols[c] for c in INPUTS], axis=-1)
        .astype(np.float32)
        .reshape(n, days, len(INPUTS)),
    )
