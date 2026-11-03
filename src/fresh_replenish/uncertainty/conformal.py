"""Calibrated forecast ranges (spec step 5): split-conformal shifts per model and category.

Each decile gets a one-sided shift learned on Phase 4's train-period folds. Errors are scaled by
the forecast's own spread (q90 - q10), so wide and narrow forecasts are adjusted in step.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fresh_replenish.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from fresh_replenish.data.db import connect, sample_stores
from fresh_replenish.data.download import local_paths
from fresh_replenish.forecasting import evaluate
from fresh_replenish.forecasting.data import HORIZON, Series, load_series
from fresh_replenish.imputation.recovery import recover_eval
from fresh_replenish.metrics import DECILE_COLUMNS, DECILES, SEED, wql
from fresh_replenish.report import md_table

TARGET, TOLERANCE = 0.8, 0.02  # the 10-90% range should cover 80% +/- 2 points
KEYS = ["store_id", "product_id"]
CALIBRATED = [f"cal_{c}" for c in DECILE_COLUMNS]
SHIFTS = [f"{c}_shift" for c in DECILE_COLUMNS]
MIN_SERIES = 10  # smaller categories share one pooled group
POOLED = -1


def group_table(ids: pd.DataFrame, min_series: int = MIN_SERIES) -> pd.DataFrame:
    """Calibration group per series: its first-level category, or POOLED when that is small."""
    category = ids.first_category_id
    small = category.map(category.value_counts()) < min_series
    return ids[KEYS].assign(group=category.where(~small, POOLED))


def _spread(frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(frame.q90 - frame.q10, 1e-6).to_numpy()[:, None]


def shifts(calibration: pd.DataFrame, group_of: pd.DataFrame) -> pd.DataFrame:
    """Conformal shift per model, group and decile, in units of forecast spread (wide table)."""
    frame = calibration.dropna(subset=DECILE_COLUMNS).merge(group_of, on=KEYS)
    rows = []
    for (model, group), part in frame.groupby(["model", "group"]):
        errors = part.actual.to_numpy()[:, None] - part[DECILE_COLUMNS].to_numpy()
        scores = np.sort(errors / _spread(part), axis=0)  # each decile's column sorted
        n = len(part)
        rank = np.minimum(np.ceil((n + 1) * DECILES).astype(int), n) - 1  # ceil((n+1)tau)-th
        shift = dict(zip(DECILE_COLUMNS, scores[rank, np.arange(len(DECILES))], strict=True))
        rows.append({"model": model, "group": group, **shift})
    return pd.DataFrame(rows)


def calibrate(
    forecasts: pd.DataFrame, shift_table: pd.DataFrame, group_of: pd.DataFrame
) -> pd.DataFrame:
    """forecasts plus calibrated deciles cal_q10 ... cal_q90 (NaN for point forecasts)."""
    frame = forecasts.merge(group_of, on=KEYS, how="left").merge(
        shift_table, on=["model", "group"], how="left", suffixes=("", "_shift")
    )
    adjusted = frame[DECILE_COLUMNS].to_numpy() + frame[SHIFTS].to_numpy() * _spread(frame)
    frame[CALIBRATED] = np.sort(np.maximum(adjusted, 0), axis=1)
    return frame.drop(columns=SHIFTS)


def _coverage(frame: pd.DataFrame, cols: list[str]) -> float:
    return float(frame.actual.between(frame[cols[0]], frame[cols[-1]]).mean())


def _wql(frame: pd.DataFrame, cols: list[str]) -> float:
    return wql(frame[cols].to_numpy()[:, :, None], frame.actual.to_numpy()[:, None])


def run(
    series: Series, folds: pd.DataFrame, forecasters: dict = evaluate.MODELS
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Forecast the held-out week (the last 7 days), calibrate on the folds, check coverage."""
    held_out, forecasts = evaluate.run(series, forecasters, origins=(len(series.dates) - HORIZON,))
    group_of = group_table(series.ids)
    table = shifts(folds, group_of)
    frame = calibrate(forecasts, table, group_of)
    rows, hits = [], []
    for model, part in frame.dropna(subset=CALIBRATED).groupby("model", sort=False):
        after = _coverage(part, CALIBRATED)
        rows.append({
            "model": model,
            "coverage_before": _coverage(part, DECILE_COLUMNS),
            "coverage_after": after,
            "within_target": abs(after - TARGET) <= TOLERANCE,
            "wql_before": _wql(part, DECILE_COLUMNS),
            "wql_after": _wql(part, CALIBRATED),
        })
        below = {c: float((part.actual <= part[f"cal_{c}"]).mean()) for c in DECILE_COLUMNS}
        hits.append({"model": model, **below})
    sample = pd.DataFrame([{
        "stores": series.ids.store_id.nunique(),
        "series": len(series.ids),
        "calibration_rows_per_model": len(folds) // folds.model.nunique(),
        "held_out_start": series.dates[-HORIZON],
        "held_out_end": series.dates[-1],
    }])
    results = {
        "coverage": pd.DataFrame(rows),
        "deciles": pd.DataFrame(hits),
        "held_out": held_out["methods"],
        "shifts": table,
        "sample": sample,
        "timing": held_out["timing"],
    }
    return results, frame


def render_report(results: dict[str, pd.DataFrame]) -> str:
    c = results["coverage"].set_index("model")
    held = results["held_out"].set_index("model")
    s = results["sample"].iloc[0]
    best = held.wape.idxmin()
    within = c.within_target.to_numpy(dtype=bool)

    def names(mask):
        return ", ".join(c.index[mask]) or "none"

    def span(col):
        return f"{c[col].min():.1%} to {c[col].max():.1%}"

    parts = [
        "# Calibrated forecast ranges",
        "Each model's deciles were shifted per product category with split-conformal "
        "calibration: a one-sided shift per decile, learned on Phase 4's "
        f"{int(s.calibration_rows_per_model):,} train-period fold forecasts per model and scaled "
        "by each forecast's own spread (q90 - q10). The check is the held-out week "
        f"({s.held_out_start} to {s.held_out_end}), which played no part in training or "
        f"calibration: {int(s.series):,} series in {int(s.stores)} stores, forecast from all "
        "earlier days, with demand recovered as in Phase 3. Generated by "
        "`uv run python -m fresh_replenish.uncertainty.conformal`.",
        "## Headline",
        f"- **Exit check:** after calibration, the 10-90% ranges cover {span('coverage_after')} "
        f"of held-out demand (target 80% +/- 2 points). Within the target: {names(within)}; "
        f"outside it: {names(~within)}.",
        f"- **Before calibration** they covered {span('coverage_before')}.",
        f"- **Held-out accuracy:** {best} has the lowest WAPE ({held.loc[best, 'wape']:.1%}); "
        f"seasonal naive has {held.loc[evaluate.BASELINE, 'wape']:.1%}.",
        "## Coverage",
        md_table(results["coverage"]),
        "## Hit rate of each calibrated decile",
        "Share of held-out demand at or below each decile; a calibrated q30 should sit near 30%.",
        md_table(results["deciles"]),
        "## Held-out week accuracy",
        md_table(results["held_out"]),
        "## Sample",
        md_table(results["sample"]),
        "## Time",
        md_table(results["timing"]),
    ]
    return "\n\n".join(parts) + "\n"


def main(
    raw_dir: Path = RAW_DIR, out_dir: Path = REPORTS_DIR, processed_dir: Path = PROCESSED_DIR
) -> int:
    paths = local_paths(raw_dir)
    if paths is None:
        return 2
    recovered = processed_dir / "recovered_demand.parquet"
    folds = processed_dir / "forecast_folds.parquet"
    for path, module in ((recovered, "imputation.recovery"), (folds, "forecasting.evaluate")):
        if not path.exists():
            print(
                f"{path.name} not found. Run: uv run python -m fresh_replenish.{module}",
                file=sys.stderr,
            )
            return 2
    con = connect(*paths)
    stores = sample_stores(con, evaluate.SAMPLE_SHARE, np.random.default_rng(SEED))
    held_out = processed_dir / "recovered_demand_eval.parquet"
    recover_eval(con, stores).to_parquet(held_out, index=False)
    series = load_series(con, [recovered, held_out], stores=stores)
    results, frame = run(series, pd.read_parquet(folds))
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / "tables" / f"calibration_{name}.csv", index=False)
    report = out_dir / "calibration.md"
    report.write_text(render_report(results), encoding="utf-8")
    frame.to_parquet(processed_dir / "forecast_eval.parquet", index=False)
    print(f"Report written to {report}; {len(frame):,} held-out forecasts saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
