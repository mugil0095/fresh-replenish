"""Forecasting comparison (spec step 4): rolling-origin validation of every forecaster.

Each model forecasts the next 7 days of recovered demand from 4 weekly origins in the train
period. Scores pool the folds; intervals come from resampling series, and the same resamples
pair the models.
"""

import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fresh_replenish.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from fresh_replenish.data.db import connect, sample_stores
from fresh_replenish.data.download import local_paths
from fresh_replenish.forecasting import models
from fresh_replenish.forecasting.data import HORIZON, ORIGINS, Series, load_series
from fresh_replenish.metrics import DECILES, SEED, bootstrap, coverage, mase, score, wql
from fresh_replenish.report import md_table

SAMPLE_SHARE = 0.10  # of stores: the same draw as Phase 3
BASELINE, CLASSICAL = "seasonal naive", "lightgbm"
MODELS = {
    BASELINE: models.seasonal_naive,
    CLASSICAL: models.lightgbm_quantiles,
    "chronos2": models.chronos2,
    "chronos2 without inputs": functools.partial(models.chronos2, use_inputs=False),
    "timesfm2.5": models.timesfm25,
    "toto2": models.toto2,
}
# From the model cards and technical reports; docs/decisions.md lists the sources.
PRETRAINING = pd.DataFrame(
    [
        ("chronos2", "Subsets of the Chronos datasets (last changed 2025-03) and GIFT-Eval "
         "Pretrain (2025-01), plus synthetic data", "no"),
        ("timesfm2.5", "GIFT-Eval Pretrain, Wikimedia pageviews (to 2023), Google Trends "
         "(to 2022), plus synthetic data", "no"),
        ("toto2", "Datadog observability metrics and synthetic series only", "no"),
    ],
    columns=["model", "pretraining data", "contains FreshRetailNet-50K"],
)


def _fold_frame(name: str, series: Series, q: np.ndarray, days: np.ndarray) -> pd.DataFrame:
    """One row per series and forecast day, with the deciles as q10 ... q90."""
    n, t = len(series.demand), len(days)
    frame = pd.DataFrame({
        "model": name,
        "fold": np.tile(np.arange(t) // HORIZON + 1, n),
        "store_id": np.repeat(series.ids.store_id.to_numpy(), t),
        "product_id": np.repeat(series.ids.product_id.to_numpy(), t),
        "dt": np.tile(series.dates[days], n),
        "actual": series.demand[:, days].ravel(),
        "in_stock": series.in_stock[:, days].ravel(),
    })
    for k, level in enumerate(DECILES):
        frame[f"q{round(level * 100)}"] = q[:, k].ravel()
    return frame


def _fold_sums(a: np.ndarray, folds: int) -> np.ndarray:
    """Totals per fold of a (series, folds * HORIZON) array."""
    return a.reshape(len(a), folds, HORIZON).sum(axis=(0, 2))


def run(
    series: Series, forecasters: dict = MODELS, origins=ORIGINS
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Score every forecaster on every fold; also return the fold forecasts for Phase 5."""
    days = np.concatenate([np.arange(o, o + HORIZON) for o in origins])
    actual = series.demand[:, days]
    n, folds = len(actual), len(origins)
    rows, draws, by_fold, timing, saved = [], {}, {}, [], []
    for name, forecaster in forecasters.items():
        start = time.perf_counter()
        q = np.concatenate([forecaster(series, o, HORIZON) for o in origins], axis=-1)
        models.release()  # one foundation model at a time fits the 4 GB GPU
        timing.append((name, round(time.perf_counter() - start, 1)))
        median = q[:, len(DECILES) // 2]
        point, draws[name] = bootstrap(score(median, actual, True))
        in_stock = score(median, actual, series.in_stock[:, days])
        by_fold[name] = _fold_sums(np.abs(median - actual), folds) / _fold_sums(actual, folds)
        ratios = []
        for i, origin in enumerate(origins):
            week = slice(i * HORIZON, (i + 1) * HORIZON)
            ratios.append(mase(median[:, week], actual[:, week], series.demand[:, :origin]))
        rows.append({
            "model": name,
            **point,
            "mase": np.concatenate(ratios).mean(),
            "wql": wql(q, actual),
            "coverage_80": coverage(q, actual),
            "wape_in_stock": in_stock[0].sum() / in_stock[2].sum(),
        })
        saved.append(_fold_frame(name, series, q, days))

    methods = pd.DataFrame(rows)
    wape = methods.set_index("model").wape
    for label, ref in (("best", wape.idxmin()), (CLASSICAL, CLASSICAL)):
        if ref not in draws:
            continue
        diffs = [draws[m] - draws[ref] for m in methods.model]
        methods[f"vs_{label}"] = methods.wape - wape[ref]
        methods[f"vs_{label}_low"] = [np.percentile(d, 2.5) for d in diffs]
        methods[f"vs_{label}_high"] = [np.percentile(d, 97.5) for d in diffs]

    fold_table = pd.DataFrame(by_fold, index=[f"fold {i + 1}" for i in range(folds)]).T
    naive = fold_table.loc[BASELINE]
    fold_table["beats_seasonal_naive"] = [
        "" if m == BASELINE else bool((fold_table.loc[m, naive.index] < naive).all())
        for m in fold_table.index
    ]
    sample = pd.DataFrame([{
        "stores": series.ids.store_id.nunique(), "series": n, "folds": folds,
        "horizon_days": HORIZON, "forecast_days": n * folds * HORIZON,
    }])
    results = {
        "methods": methods,
        "folds": fold_table.rename_axis("model").reset_index(),
        "sample": sample,
        "timing": pd.DataFrame(timing, columns=["model", "seconds"]),
    }
    return results, pd.concat(saved, ignore_index=True)


def render_report(results: dict[str, pd.DataFrame]) -> str:
    m = results["methods"].set_index("model")
    beats = results["folds"].set_index("model").beats_seasonal_naive.drop(BASELINE).astype(bool)
    s = results["sample"].iloc[0]
    best = m.wape.idxmin()

    def pct(row, key, spec):
        low, high = row[key + "_low"], row[key + "_high"]
        return f"{row[key]:{spec}} (95% CI {low:{spec}} to {high:{spec}})"

    def names(mask):
        return ", ".join(beats.index[mask]) or "none"

    parts = [
        "# Forecasting daily demand 7 days ahead",
        f"Each model forecast the next 7 days of recovered demand (Phase 3) from 4 weekly "
        f"origins in the train period, for {int(s.series):,} store-product series in a random "
        f"{int(s.stores)} stores: {int(s.forecast_days):,} forecast days per model. The "
        "foundation models are zero-shot; LightGBM is trained on each fold's history. WAPE is "
        "the average error as a share of demand; WPE is the bias (negative means too low); MASE "
        "compares the error with repeating last week; WQL is the scaled pinball loss of the "
        "deciles; coverage_80 is the share of actual demand inside the 10-90% range (target "
        "80%, calibrated in Phase 5). Intervals come from 1,000 bootstrap resamples of series, "
        "and the same resamples pair the models. Generated by "
        "`uv run python -m fresh_replenish.forecasting.evaluate`.",
        "## Headline",
        f"- **Best model: {best}.** WAPE {pct(m.loc[best], 'wape', '.1%')}; "
        f"bias {pct(m.loc[best], 'wpe', '+.1%')}.",
        f"- **Against seasonal naive** (WAPE {m.loc[BASELINE, 'wape']:.1%}): beat it in every "
        f"fold: {names(beats.to_numpy())}; do not: {names(~beats.to_numpy())}.",
    ]
    if "vs_lightgbm" in m:
        gaps = "; ".join(
            f"{name} {pct(row, 'vs_lightgbm', '+.1%')}"
            for name, row in m.drop([BASELINE, CLASSICAL]).iterrows()
        )
        parts.append(f"- **Against LightGBM** (paired WAPE gap, negative is better): {gaps}.")
    parts += [
        "## Models",
        md_table(results["methods"]),
        "## WAPE by fold",
        md_table(results["folds"]),
        "## Pretraining-leakage check",
        "FreshRetailNet-50K was first published on 2025-05-07. Each model's published training "
        "data was checked for it; `docs/decisions.md` lists the sources.",
        md_table(PRETRAINING),
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
    if not recovered.exists():
        print(
            "Recovered demand not found. Run: uv run python -m fresh_replenish.imputation.recovery",
            file=sys.stderr,
        )
        return 2
    con = connect(*paths)
    stores = sample_stores(con, SAMPLE_SHARE, np.random.default_rng(SEED))
    results, saved = run(load_series(con, recovered, stores=stores))
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / "tables" / f"forecast_{name}.csv", index=False)
    report = out_dir / "forecasting.md"
    report.write_text(render_report(results), encoding="utf-8")
    saved.to_parquet(processed_dir / "forecast_folds.parquet", index=False)
    print(f"Report written to {report}; {len(saved):,} fold forecasts saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
