"""Reorder API (spec step 7): serves the nightly batch's orders, one store per request.

Run it with `uv run uvicorn --factory fresh_replenish.serving.api:create_app`. Another cost
ratio is recomputed from the stored calibrated deciles, so no model runs at request time.
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

from fresh_replenish.config import PROCESSED_DIR
from fresh_replenish.decisions.newsvendor import HEADLINE, critical_ratio, order_at
from fresh_replenish.metrics import DECILE_COLUMNS

ORDERS = PROCESSED_DIR / "orders.parquet"


class Line(BaseModel):
    product_id: int
    date: str
    order: float
    low: float  # q10
    median: float  # q50
    high: float  # q90


class Reorder(BaseModel):
    store_id: int
    as_of: str
    ratio: float
    critical_ratio: float
    lines: list[Line]


def create_app(orders_path: Path = ORDERS) -> FastAPI:
    loaded = (None, {}, None)  # (file mtime, orders by store, as_of), replaced as one tuple
    app = FastAPI(title="fresh-replenish", summary="Cost-based daily orders for dark stores")

    def current() -> tuple[dict, str | None]:
        """Orders by store and their as-of date, read again whenever the batch rewrites them."""
        nonlocal loaded
        try:
            mtime = orders_path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = None
        if mtime != loaded[0]:
            orders = None if mtime is None else pd.read_parquet(orders_path)
            if orders is None or orders.empty:
                loaded = (mtime, {}, None)
            else:
                stores = dict(tuple(orders.groupby("store_id")))
                loaded = (mtime, stores, str(orders.as_of.iloc[0]))
        return loaded[1], loaded[2]

    @app.get("/health")
    def health(response: Response) -> dict:
        stores, as_of = current()
        if not stores:
            response.status_code = 503  # so a health check fails until the batch has run
        return {
            "status": "ok" if stores else "no orders", "orders_loaded": bool(stores),
            "as_of": as_of,
        }

    @app.get("/reorder")
    def reorder(
        store_id: int,
        ratio: float = Query(HEADLINE, ge=1 / 9, le=9, description="lost-sale cost / waste cost"),
    ) -> Reorder:
        stores, as_of = current()
        if not stores:
            raise HTTPException(503, "no orders yet: run the batch job")
        rows = stores.get(store_id)
        if rows is None:
            raise HTTPException(404, f"no orders for store {store_id}")
        q = rows[DECILE_COLUMNS].to_numpy()
        level = critical_ratio(ratio)
        columns = (rows.product_id, rows.dt, order_at(q, level), q[:, 0], q[:, 4], q[:, -1])
        lines = [
            Line(product_id=p, date=d, order=o, low=low, median=mid, high=high)
            for p, d, o, low, mid, high in zip(*(list(c) for c in columns), strict=True)
        ]
        return Reorder(
            store_id=store_id, as_of=as_of, ratio=ratio, critical_ratio=level, lines=lines
        )

    return app

