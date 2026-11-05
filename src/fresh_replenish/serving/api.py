"""Reorder API (spec step 7): serves the nightly batch's orders, one store per request.

Run it with `uv run uvicorn --factory fresh_replenish.serving.api:create_app`. Another cost
ratio is recomputed from the stored calibrated deciles, so no model runs at request time. A store
with a second delivery posts its midday shelf count to /topup for the units to order.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from fresh_replenish.config import PROCESSED_DIR
from fresh_replenish.decisions.newsvendor import HEADLINE, critical_ratio, order_at, top_up
from fresh_replenish.metrics import DECILE_COLUMNS, REST_COLUMNS

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
    morning_share: float | None = None  # with a second delivery; the 9x plan at every ratio
    lines: list[Line]


class ShelfCount(BaseModel):
    store_id: int
    date: str
    shelf: dict[int, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(min_length=1)
    ratio: float = Field(HEADLINE, ge=1 / 9, le=9, description="lost-sale cost / waste cost")


class TopUpLine(BaseModel):
    product_id: int
    shelf: float
    target: float
    top_up: float


class TopUp(BaseModel):
    store_id: int
    date: str
    as_of: str
    ratio: float
    critical_ratio: float
    decision_hour: int  # the targets cover the demand left from this hour on
    arrival_hour: int
    lines: list[TopUpLine]


def create_app(orders_path: Path = ORDERS) -> FastAPI:
    loaded = (None, {}, None)  # (file mtime, orders by store, as_of), replaced as one tuple
    app = FastAPI(title="fresh-replenish", summary="Cost-based daily orders for dark stores")

    @app.exception_handler(RequestValidationError)
    def refuse(_: Request, error: RequestValidationError) -> JSONResponse:
        """FastAPI's 422 without echoing the input: a NaN count cannot be written back as JSON."""
        fields = ("type", "loc", "msg")
        return JSONResponse(
            {"detail": [{k: e[k] for k in fields} for e in error.errors()]}, status_code=422
        )

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

    def store_rows(store_id: int) -> tuple[pd.DataFrame, str]:
        stores, as_of = current()
        if not stores:
            raise HTTPException(503, "no orders yet: run the batch job")
        rows = stores.get(store_id)
        if rows is None:
            raise HTTPException(404, f"no orders for store {store_id}")
        return rows, as_of

    @app.get("/reorder")
    def reorder(
        store_id: int,
        ratio: float = Query(HEADLINE, ge=1 / 9, le=9, description="lost-sale cost / waste cost"),
    ) -> Reorder:
        rows, as_of = store_rows(store_id)
        q = rows[DECILE_COLUMNS].to_numpy()
        level = critical_ratio(ratio)
        columns = (rows.product_id, rows.dt, order_at(q, level), q[:, 0], q[:, 4], q[:, -1])
        lines = [
            Line(product_id=p, date=d, order=o, low=low, median=mid, high=high)
            for p, d, o, low, mid, high in zip(*(list(c) for c in columns), strict=True)
        ]
        share = rows.morning_share.iloc[0] if "morning_share" in rows else None
        return Reorder(
            store_id=store_id, as_of=as_of, ratio=ratio, critical_ratio=level,
            morning_share=share, lines=lines,
        )

    @app.post("/topup")
    def count_shelf(count: ShelfCount) -> TopUp:
        rows, as_of = store_rows(count.store_id)
        if REST_COLUMNS[0] not in rows:
            raise HTTPException(503, "no top-up targets yet: run the batch job")
        day = rows[rows.dt == count.date].set_index("product_id")
        if day.empty:
            raise HTTPException(404, f"no orders for store {count.store_id} on {count.date}")
        unknown = sorted(set(count.shelf) - set(day.index))
        if unknown:
            raise HTTPException(422, f"store {count.store_id} has no orders for {unknown}")
        products, shelf = list(count.shelf), np.array(list(count.shelf.values()))
        level = critical_ratio(count.ratio)
        target = order_at(day.loc[products, REST_COLUMNS].to_numpy(), level)
        lines = [
            TopUpLine(product_id=p, shelf=s, target=t, top_up=u)
            for p, s, t, u in zip(products, shelf, target, top_up(target, shelf), strict=True)
        ]
        return TopUp(
            store_id=count.store_id, date=count.date, as_of=as_of, ratio=count.ratio,
            critical_ratio=level, decision_hour=day.decision_hour.iloc[0],
            arrival_hour=day.arrival_hour.iloc[0], lines=lines,
        )

    return app

