"""Nightly batch (spec step 7): calibrated, cost-based orders for every series for the next week.

It runs "as of" the last train day, so the week it orders for is the held-out week and monitoring
can check it later. LightGBM, the Phase 4-5 winner, trains on recovered demand; Phase 5's
conformal shifts calibrate its deciles; the newsvendor rule sets each order.
"""

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fresh_replenish.config import PROCESSED_DIR, RAW_DIR, REPO_ROOT, REPORTS_DIR
from fresh_replenish.data.db import check_grid, connect
from fresh_replenish.data.download import local_paths
from fresh_replenish.decisions.newsvendor import HEADLINE, critical_ratio, order_at
from fresh_replenish.forecasting import models
from fresh_replenish.forecasting.data import CLASSICAL, HORIZON, INPUTS, Series, load_series
from fresh_replenish.forecasting.evaluate import forecast_frame
from fresh_replenish.metrics import CALIBRATED_COLUMNS, DECILE_COLUMNS
from fresh_replenish.uncertainty.conformal import POOLED, calibrate

TRACKING_DIR = REPO_ROOT / "mlruns"  # git-ignored local MLflow store


def with_future(con: duckdb.DuckDBPyConnection, series: Series) -> Series:
    """The series plus the eval week's known inputs, with demand left blank."""
    stores = sorted(series.ids.store_id.unique().tolist())
    cols = con.execute(
        f"SELECT store_id, product_id, dt, {', '.join(INPUTS)} FROM eval "
        "WHERE list_contains($stores::BIGINT[], store_id) ORDER BY store_id, product_id, dt",
        {"stores": stores},
    ).fetchnumpy()
    days, dates = check_grid(cols)
    keys = np.column_stack([cols["store_id"], cols["product_id"]])[::days]
    if not np.array_equal(keys, series.ids[["store_id", "product_id"]].to_numpy()):
        raise ValueError("the eval week must cover exactly the same series")
    n = len(keys)
    inputs = np.stack([cols[c] for c in INPUTS], axis=-1).astype(np.float32)
    return Series(
        ids=series.ids,
        dates=np.concatenate([series.dates, dates]),
        demand=np.concatenate([series.demand, np.full((n, days), np.nan, np.float32)], axis=1),
        in_stock=np.concatenate([series.in_stock, np.zeros((n, days), dtype=bool)], axis=1),
        inputs=np.concatenate([series.inputs, inputs.reshape(n, days, len(INPUTS))], axis=1),
    )


def forecast_orders(
    series: Series, origin: int, shifts: pd.DataFrame, forecaster, ratio: float = HEADLINE
) -> pd.DataFrame:
    """Calibrated deciles (q10 ... q90) and the order at the cost ratio for the week from origin."""
    days = np.arange(origin, origin + HORIZON)
    frame = forecast_frame(CLASSICAL, series, forecaster(series, origin, HORIZON), days)
    category = series.ids.first_category_id
    known = category.isin(shifts.loc[shifts.model == CLASSICAL, "group"])
    group_of = series.ids[["store_id", "product_id"]].assign(group=category.where(known, POOLED))
    frame = calibrate(frame, shifts, group_of)
    frame["order"] = order_at(frame[CALIBRATED_COLUMNS].to_numpy(), critical_ratio(ratio))
    frame = frame.assign(as_of=series.dates[origin - 1], ratio=ratio)
    out = frame[["store_id", "product_id", "dt", *CALIBRATED_COLUMNS, "order", "as_of", "ratio"]]
    return out.rename(columns=dict(zip(CALIBRATED_COLUMNS, DECILE_COLUMNS, strict=True)))


def log_run(tracking_dir: Path, name: str, params: dict, metrics: dict) -> None:
    """One MLflow run, stored in SQLite under tracking_dir (MLflow 3 retired the file store)."""
    import mlflow

    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("fresh-replenish")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def main(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    out_dir: Path = REPORTS_DIR,
    tracking_dir: Path = TRACKING_DIR,
) -> int:
    paths = local_paths(raw_dir)
    if paths is None:
        return 2
    recovered = processed_dir / "recovered_demand.parquet"
    shifts = out_dir / "tables" / "calibration_shifts.csv"
    for path, module in ((recovered, "imputation.recovery"), (shifts, "uncertainty.conformal")):
        if not path.exists():
            print(
                f"{path.name} not found. Run: uv run python -m fresh_replenish.{module}",
                file=sys.stderr,
            )
            return 2
    start = time.perf_counter()
    con = connect(*paths)
    series = load_series(con, recovered)
    orders = forecast_orders(
        with_future(con, series), len(series.dates), pd.read_csv(shifts), models.lightgbm_quantiles
    )
    runtime = round(time.perf_counter() - start, 1)
    orders.to_parquet(processed_dir / "orders.parquet", index=False)
    summary = {
        "as_of": orders.as_of.iloc[0], "model": CLASSICAL, "ratio": HEADLINE,
        "series": len(series.ids), "order_lines": len(orders), "runtime_s": runtime,
    }
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_dir / "tables" / "service_batch.csv", index=False)
    log_run(
        tracking_dir, "batch",
        {k: summary[k] for k in ("as_of", "model", "ratio", "series")},
        {"runtime_s": runtime, "order_units": float(orders.order.sum())},
    )
    print(f"{len(orders):,} order lines for {len(series.ids):,} series in {runtime} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
