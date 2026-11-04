"""Data for the dashboard: where money is lost, the headline, forecasts and decision records."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fresh_replenish.decisions.newsvendor import HEADLINE, critical_ratio, order_at
from fresh_replenish.forecasting.data import BASELINE, CLASSICAL
from fresh_replenish.metrics import CALIBRATED_COLUMNS


def losses(
    forecasts: pd.DataFrame, categories: pd.DataFrame, ratio: float = HEADLINE,
    model: str = CLASSICAL, by: str = "first_category_id",
) -> pd.DataFrame:
    """Waste and lost-sales cost of the model's cost-based orders by group, costliest first.

    A wasted unit costs 1 and a lost sale costs `ratio`, as in the policy simulation.
    """
    rows = forecasts[forecasts.model == model].merge(categories, on=["store_id", "product_id"])
    order = order_at(rows[CALIBRATED_COLUMNS].to_numpy(), critical_ratio(ratio))
    rows = rows.assign(
        waste_cost=np.maximum(order - rows.actual, 0),
        lost_sales_cost=ratio * np.maximum(rows.actual - order, 0),
    )
    out = rows.groupby(by)[["waste_cost", "lost_sales_cost", "actual"]].sum()
    out["total"] = out.waste_cost + out.lost_sales_cost
    return out.sort_values("total", ascending=False).reset_index()


def headline(policies: pd.DataFrame, ratio: float = HEADLINE) -> dict:
    """Cost-based orders against the most accurate model's median and against last week's demand."""
    at = policies[policies.ratio == ratio].set_index("policy").cost
    median = next(p for p in at.index if p.endswith(" median"))
    model = median.removesuffix(" median")
    return {
        "model": model, "ratio": ratio, "cost": at[model],
        "vs_median": 1 - at[model] / at[median], "vs_naive": 1 - at[model] / at[BASELINE],
    }


def categories(train_path: Path) -> pd.DataFrame:
    """First-level category of every store-product series."""
    return duckdb.sql(
        "SELECT DISTINCT store_id, product_id, first_category_id FROM read_parquet($path)",
        params={"path": str(train_path)},
    ).df()


def history(recovered_path: Path, store_id: int, product_id: int, days: int = 28) -> pd.DataFrame:
    """The last `days` days of one series' recovered demand."""
    return duckdb.sql(
        "SELECT dt, recovered_demand FROM read_parquet($path) "
        "WHERE store_id = $store AND product_id = $product ORDER BY dt DESC LIMIT $days",
        params={"path": str(recovered_path), "store": store_id, "product": product_id,
                "days": days},
    ).df().iloc[::-1]


def records(path: Path) -> pd.DataFrame:
    """The agent's decision records, newest first."""
    columns = ["time", "store_id", "product_id", "date", "order", "reason"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows, columns=columns).sort_values("time", ascending=False)
