"""Tiny pipeline outputs, so CI can run the real dashboard and API without the dataset.

`python tests/dashboard/smoke_data.py data` writes them where `docker compose up` reads them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fresh_replenish.forecasting.data import CLASSICAL
from fresh_replenish.metrics import CALIBRATED_COLUMNS, DECILE_COLUMNS, REST_COLUMNS

STORE, PRODUCT, DAY = 1, 10, "2024-06-26"
COUNT = {"store_id": STORE, "date": DAY, "shelf": {str(PRODUCT): 2}}


def write(root: Path) -> None:
    """Write one order line's outputs under `root`, laid out like `data/`."""
    processed, raw = root / "processed", root / "raw" / "data"
    processed.mkdir(parents=True)  # no exist_ok: this must never replace real pipeline outputs
    raw.mkdir(parents=True)
    line = pd.DataFrame({"store_id": [STORE], "product_id": [PRODUCT], "dt": [DAY]})
    deciles = np.arange(1.0, 10.0)
    orders = line.assign(as_of="2024-06-25", decision_hour=13, arrival_hour=15, morning_share=0.8)
    orders[DECILE_COLUMNS + REST_COLUMNS] = np.r_[deciles, deciles / 2]
    orders.to_parquet(processed / "orders.parquet")
    forecasts = line.assign(model=CLASSICAL, actual=4.0)
    forecasts[CALIBRATED_COLUMNS] = deciles
    forecasts.to_parquet(processed / "forecast_eval.parquet")
    line.assign(recovered_demand=4.0).to_parquet(processed / "recovered_demand.parquet")
    line.assign(first_category_id=5).to_parquet(raw / "train.parquet")


if __name__ == "__main__":
    write(Path(sys.argv[1]))
