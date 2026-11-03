"""Hidden-demand recovery (spec step 3): score four methods on hidden in-stock hours.

Protocol from the dataset paper (Dingdong-Inc/frn-50k-baseline, `evaluation_mnar`): hide
in-stock hours with `mimic_missing`, then compare daily totals on days that got hidden hours
but had no real stockout, where the true demand is known.
"""

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fresh_replenish.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from fresh_replenish.data.db import connect, sample_stores
from fresh_replenish.data.download import local_paths
from fresh_replenish.imputation.data import load_hours
from fresh_replenish.imputation.deep import fit_deep, predict_deep
from fresh_replenish.imputation.masking import mimic_missing
from fresh_replenish.imputation.statistical import profile_recover, tobit_recover
from fresh_replenish.metrics import bootstrap, score
from fresh_replenish.report import md_table

SEED = 42
SAMPLE_SHARE = 0.10  # of stores; keeps SAITS and TimesNet inside the GTX 1650's 4 GB
STATISTICAL = ["profile", "tobit"]
DEEP = ["saits", "timesnet"]


def _recover(name: str, sales, covariates, deep: dict) -> np.ndarray:
    """Daily operating-hour demand (series, days) from one method."""
    if name == "profile":
        return profile_recover(sales).sum(axis=-1)
    if name == "tobit":
        return tobit_recover(sales, covariates)
    return predict_deep(deep[name], sales, covariates).sum(axis=-1)


def run(
    con: duckdb.DuckDBPyConnection, sample_share: float = SAMPLE_SHARE, epochs: int = 10
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Score every method on a store sample, then recover all series with the chosen one."""
    rng = np.random.default_rng(SEED)
    stores = sample_stores(con, sample_share, rng)
    h = load_hours(con, stores=stores)
    n, days, hours = h.sales.shape
    masked, valid = mimic_missing(h.sales.reshape(n * 3, days // 3, hours), rng)
    masked, valid = masked.reshape(h.sales.shape), valid.reshape(h.sales.shape)
    valid_days = valid.any(axis=-1) & (h.stockout_hours == 0)
    if not valid_days.any():
        raise ValueError("no hidden in-stock day to score; use a larger sample_share")

    operating, deep, timing = {"no recovery": np.nansum(masked, axis=-1)}, {}, []
    for name in STATISTICAL + DEEP:
        start = time.perf_counter()
        if name in DEEP:
            deep[name] = fit_deep(masked, h.covariates, name, epochs=epochs)
        operating[name] = _recover(name, masked, h.covariates, deep)
        timing.append((name, round(time.perf_counter() - start, 1)))

    rows, draws = [], {}
    for name, op in operating.items():
        point, draws[name] = bootstrap(score(op + h.night, h.sale_amount, valid_days))
        rows.append({"method": name, **point})
    methods = pd.DataFrame(rows)
    best = methods[methods.method != "no recovery"].sort_values("wape").method.iloc[0]
    diffs = [draws[m] - draws[best] for m in methods.method]
    methods["vs_best"] = methods.wape - methods.set_index("method").wape[best]
    methods["vs_best_low"] = [np.percentile(d, 2.5) for d in diffs]
    methods["vs_best_high"] = [np.percentile(d, 97.5) for d in diffs]

    chosen, reason = best, "lowest WAPE"
    if best in DEEP:
        stat = methods[methods.method.isin(STATISTICAL)].sort_values("wape").method.iloc[0]
        low, high = np.percentile(draws[best] - draws[stat], [2.5, 97.5])
        if low <= 0 <= high:
            chosen, reason = stat, f"{best} is not clearly better than {stat} (interval spans 0)"

    full = load_hours(con)  # all series; blanks are real stockouts only
    start = time.perf_counter()
    demand = _recover(chosen, full.sales, full.covariates, deep) + full.night
    timing.append((f"recover all series ({chosen})", round(time.perf_counter() - start, 1)))
    n_all = len(full.ids)
    recovered = pd.DataFrame({
        "store_id": np.repeat(full.ids.store_id.to_numpy(), days),
        "product_id": np.repeat(full.ids.product_id.to_numpy(), days),
        "dt": np.tile(full.dates, n_all),
        "sale_amount": full.sale_amount.ravel(),
        "stockout_hours": full.stockout_hours.ravel(),
    })
    in_stock = recovered.stockout_hours.to_numpy() == 0
    # Demand is never below what was sold: a stockout hour can still record sales.
    recovered["recovered_demand"] = np.where(
        in_stock, recovered.sale_amount, np.maximum(demand.ravel(), recovered.sale_amount)
    )

    sample = pd.DataFrame([{
        "stores": len(stores), "series": n, "valid_days": int(valid_days.sum()),
        "hidden_hour_share": float(valid.mean()), "epochs": epochs,
    }])
    results = {
        "methods": methods,
        "choice": pd.DataFrame([{"chosen": chosen, "reason": reason}]),
        "sample": sample,
        "timing": pd.DataFrame(timing, columns=["step", "seconds"]),
    }
    return results, recovered


def render_report(results: dict[str, pd.DataFrame]) -> str:
    m = results["methods"].set_index("method")
    s = results["sample"].iloc[0]
    choice = results["choice"].iloc[0]
    best = m.drop("no recovery").wape.idxmin()

    def pct(row, key, spec):
        low, high = row[key + "_low"], row[key + "_high"]
        return f"{row[key]:{spec}} (95% CI {low:{spec}} to {high:{spec}})"

    parts = [
        "# Recovering demand hidden by stockouts",
        f"Using the dataset paper's own masking, we hid {s.hidden_hour_share:.1%} of hours in a "
        f"random sample of {int(s.stores)} stores ({int(s.series):,} store-product series). We "
        f"then scored daily totals on the {int(s.valid_days):,} days that had hidden hours but "
        "no real stockout, "
        "where the true demand is known. WAPE is the average error as a share of demand; WPE is "
        "the bias, and negative means demand is understated. Intervals come from 1,000 bootstrap "
        "resamples of series. Generated by `uv run python -m fresh_replenish.imputation.recovery`.",
        "## Headline",
        f"- **Without recovery,** demand on those days is understated by "
        f"{-m.loc['no recovery', 'wpe']:.1%}.",
        f"- **Best method: {best}.** WAPE {pct(m.loc[best], 'wape', '.1%')}; "
        f"bias {pct(m.loc[best], 'wpe', '+.1%')}.",
        f"- **Used for Phase 4:** {choice.chosen} ({choice.reason}). Recovered demand for all "
        "series is in `data/processed/recovered_demand.parquet`.",
        "## Methods",
        md_table(results["methods"]),
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
    results, recovered = run(connect(*paths))
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / "tables" / f"recovery_{name}.csv", index=False)
    report = out_dir / "demand_recovery.md"
    report.write_text(render_report(results), encoding="utf-8")
    processed_dir.mkdir(parents=True, exist_ok=True)
    recovered.to_parquet(processed_dir / "recovered_demand.parquet", index=False)
    print(f"Report written to {report}; {len(recovered):,} recovered rows saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
