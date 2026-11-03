"""The agent's tools (spec section 11): plain code, no LLM inside.

Numbers come from the nightly batch's calibrated forecasts (`data/processed/orders.parquet`).
The limits are assumed values: the data has no prices or store capacities.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fresh_replenish.config import REPO_ROOT
from fresh_replenish.decisions.newsvendor import HEADLINE, critical_ratio, order_at
from fresh_replenish.forecasting.data import HORIZON, INPUTS, Series, load_series
from fresh_replenish.metrics import DECILE_COLUMNS

LINE_FACTOR = 1.5  # an order above 1.5 x the item's 90th-percentile forecast needs a human
STORE_DAY_CAPACITY = 400.0  # units per store per day; planned days reach 366 at the 99th pct
RECORDS = REPO_ROOT / "data" / "decisions" / "records.jsonl"
NOTE = "model-based estimate (Chronos-2), not proof of cause and effect"

# (store_id, product_id, date, {input: value}) -> (planned median, changed median) or None
Scenario = Callable[[int, int, str, dict], tuple[float, float] | None]


def _r(x: float, places: int = 2) -> float:
    return round(float(x), places)


class Toolbox:
    def __init__(
        self,
        orders: pd.DataFrame,
        scenario: Scenario | None = None,
        records: Path = RECORDS,
        capacity: float = STORE_DAY_CAPACITY,
    ):
        self.orders = orders.set_index(["store_id", "product_id", "dt"]).sort_index()
        self.store_day = orders.groupby(["store_id", "dt"])["order"].sum()
        self.scenario, self.records, self.capacity = scenario, records, capacity

    def _line(self, store_id: int, product_id: int, date: str) -> pd.Series | None:
        key = (store_id, product_id, date)
        return self.orders.loc[key] if key in self.orders.index else None

    @staticmethod
    def _missing(store_id: int, product_id: int, date: str) -> dict:
        return {"error": f"no forecast for store {store_id}, product {product_id} on {date}"}

    def get_forecast(self, store_id: int, product_id: int, date: str) -> dict:
        """Calibrated forecast range (10th to 90th percentile) and median for one item-day."""
        line = self._line(store_id, product_id, date)
        if line is None:
            return self._missing(store_id, product_id, date)
        return {
            "store_id": store_id, "product_id": product_id, "date": date, "as_of": line.as_of,
            "low": _r(line.q10), "median": _r(line.q50), "high": _r(line.q90),
        }

    def compute_order(
        self, store_id: int, product_id: int, date: str, ratio: float = HEADLINE,
        units: float | None = None,
    ) -> dict:
        """Cost-based order at the ratio (or `units` if given), checked against the limits."""
        line = self._line(store_id, product_id, date)
        if line is None:
            return self._missing(store_id, product_id, date)
        if not 1 / 9 <= ratio <= 9:
            return {"error": "ratio (lost-sale cost / waste cost) must be between 0.11 and 9"}
        level = critical_ratio(ratio)
        cost_based = float(order_at(line[DECILE_COLUMNS].to_numpy(dtype=float), level))
        order = cost_based if units is None else float(units)
        line_limit = LINE_FACTOR * line.q90
        store_total = self.store_day[(store_id, date)] - line.order + order
        return {
            "store_id": store_id, "product_id": product_id, "date": date,
            "ratio": ratio, "critical_ratio": _r(level), "cost_based_order": _r(cost_based),
            "order": _r(order), "line_limit": _r(line_limit),
            "store_day_total": _r(store_total), "store_day_capacity": self.capacity,
            "within_limits": bool(order <= line_limit and store_total <= self.capacity),
        }

    def what_if(
        self, store_id: int, product_id: int, date: str, discount: float | None = None,
        promotion: int | None = None, holiday: int | None = None,
    ) -> dict:
        """Median forecast with changed inputs against the planned ones, for one item-day."""
        changes = {
            name: value
            for name, value in zip(INPUTS, (discount, promotion, holiday), strict=True)
            if value is not None
        }
        if not changes:
            return {"error": "give at least one change: discount, promotion or holiday"}
        result = self.scenario and self.scenario(store_id, product_id, date, changes)
        if result is None:
            return {"error": f"no scenario for store {store_id}, product {product_id} on {date}"}
        base, changed = result
        return {
            "store_id": store_id, "product_id": product_id, "date": date, "changes": changes,
            "planned_median": _r(base), "scenario_median": _r(changed),
            "change_pct": _r(100 * (changed / base - 1), 1) if base > 0 else None, "note": NOTE,
        }

    def write_decision_record(
        self, store_id: int, product_id: int, date: str, order: float, reason: str,
        tool_outputs: list,
    ) -> dict:
        """Append the order, its reason and the conversation's tool outputs to the record file."""
        record = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"), "store_id": store_id,
            "product_id": product_id, "date": date, "order": order, "reason": reason,
            "tool_outputs": tool_outputs,
        }
        self.records.parent.mkdir(parents=True, exist_ok=True)
        with self.records.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        return {"saved": True}


def chronos_scenario(con: duckdb.DuckDBPyConnection, recovered: Path, forecaster=None) -> Scenario:
    """What-ifs from Chronos-2: the batch's week, one item, one input changed on one day."""
    from fresh_replenish.forecasting import models
    from fresh_replenish.serving.batch import with_future

    forecaster = forecaster or models.chronos2

    def run(store_id: int, product_id: int, date: str, changes: dict):
        train = load_series(con, recovered, stores=[store_id])
        series = with_future(con, train)
        rows = np.flatnonzero(series.ids.product_id.to_numpy() == product_id)
        day = np.flatnonzero(series.dates == date)
        origin = len(train.dates)
        if not len(rows) or not len(day) or not 0 <= day[0] - origin < HORIZON:
            return None
        pair = np.repeat(rows[:1], 2)  # row 0 keeps the planned inputs, row 1 gets the change
        two = Series(
            ids=series.ids.iloc[pair].reset_index(drop=True), dates=series.dates,
            demand=series.demand[pair], in_stock=series.in_stock[pair],
            inputs=series.inputs[pair].copy(),
        )
        for name, value in changes.items():
            two.inputs[1, day[0], INPUTS.index(name)] = value
        median = forecaster(two, origin, HORIZON)[:, len(DECILE_COLUMNS) // 2, day[0] - origin]
        return float(median[0]), float(median[1])

    return run
