"""What drives daily demand (spec step 2): Poisson fixed-effects regression (PPML)."""

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

from fresh_replenish.config import DATASET_FILES, RAW_DIR, REPORTS_DIR
from fresh_replenish.data.db import connect
from fresh_replenish.report import md_table

TERMS = [
    "log_discount",
    "activity_flag",
    "holiday_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]

# discount = 0 rows are excluded: ln(0) is undefined and they sell ~15x normal (see decisions.md).
FRAME_SQL = """
SELECT store_id, product_id, city_id, sale_amount,
       ln(discount) AS log_discount, activity_flag, holiday_flag,
       precpt, avg_temperature, avg_humidity, avg_wind_level,
       strftime(date_trunc('week', CAST(dt AS DATE)), '%Y-%m-%d') AS week
FROM daily
WHERE split = 'train' AND discount > 0
  AND (NOT $in_stock_only OR stockout_hours = 0)
  AND NOT list_contains($exclude::BIGINT[], city_id)
"""


def build_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    in_stock_only: bool = False,
    exclude_cities: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Narrow model frame (no hourly lists) for the train split."""
    params = {"in_stock_only": in_stock_only, "exclude": list(exclude_cities)}
    return con.execute(FRAME_SQL, params).df()


# Week FE (not date FE) keep the date-level holiday flag identifiable.
FORMULA = f"sale_amount ~ {' + '.join(TERMS)} | store_id^product_id + week"


def fit_drivers(frame: pd.DataFrame) -> pd.DataFrame:
    """PPML coefficients with store-clustered standard errors, indexed by term."""
    tidy = pf.fepois(FORMULA, data=frame, vcov={"CRV1": "store_id"}).tidy()
    out = pd.DataFrame(
        {"coef": tidy["Estimate"], "se": tidy["Std. Error"], "p": tidy["Pr(>|t|)"]}
    )
    out.index.name = "term"
    return out


def split_stores(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two halves of the stores, chosen at random (same seed, same split)."""
    stores = np.sort(frame.store_id.unique())
    half = np.random.default_rng(seed).choice(stores, size=len(stores) // 2, replace=False)
    in_a = frame.store_id.isin(half)
    return frame[in_a], frame[~in_a]


def stability(fit_a: pd.DataFrame, fit_b: pd.DataFrame) -> pd.DataFrame:
    """Stable = same sign in both halves and a difference under 1.96 combined standard errors."""
    table = fit_a[["coef", "se"]].join(
        fit_b[["coef", "se"]], lsuffix="_a", rsuffix="_b", how="outer"
    ).reindex(TERMS)
    table["z_diff"] = (table.coef_a - table.coef_b) / np.sqrt(table.se_a**2 + table.se_b**2)
    table["same_sign"] = np.sign(table.coef_a) == np.sign(table.coef_b)
    table["stable"] = table.same_sign & (table.z_diff.abs() < 1.96)  # a missing term (NaN) is False
    return table


MEANING = {
    "log_discount": "10% price cut",
    "activity_flag": "promotion running",
    "holiday_flag": "non-working day",
    "precpt": "+1 mm rain",
    "avg_temperature": "+1 degree C",
    "avg_humidity": "+1 point humidity",
    "avg_wind_level": "+1 wind level",
}


def effects(fit: pd.DataFrame) -> pd.DataFrame:
    """% change in sales for the change named in `meaning`, with a 95% CI."""
    # PPML: sales scale with exp(coef * x); a 10% price cut moves log price by ln(0.9).
    step = pd.Series([np.log(0.9) if t == "log_discount" else 1.0 for t in fit.index], fit.index)

    def pct(coef: pd.Series) -> pd.Series:
        return (np.exp(coef * step) - 1) * 100

    low, high = pct(fit.coef - 1.96 * fit.se), pct(fit.coef + 1.96 * fit.se)
    return pd.DataFrame(
        {
            "term": fit.index,
            "meaning": fit.index.map(MEANING),
            "effect_pct": pct(fit.coef).values,
            "ci_low_pct": np.minimum(low, high).values,  # a price cut flips the bound order
            "ci_high_pct": np.maximum(low, high).values,
            "p": fit.p.values,
        }
    )


ROBUSTNESS = {
    "in-stock days only": {"in_stock_only": True},
    "without city 0": {"exclude_cities": (0,)},
}


def run(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Main fit, two random store halves and two robustness fits, with timings."""
    fits, timing = {}, []

    def fit(name: str, frame: pd.DataFrame) -> None:
        start = time.perf_counter()
        fits[name] = fit_drivers(frame)
        timing.append((name, len(frame), round(time.perf_counter() - start, 1)))

    # Build one frame (or one pair) at a time so peak memory stays near a single frame.
    main_frame = build_frame(con)
    fit("all stores", main_frame)
    half_a, half_b = split_stores(main_frame)
    del main_frame
    fit("half A", half_a)
    fit("half B", half_b)
    del half_a, half_b
    for name, kwargs in ROBUSTNESS.items():
        fit(name, build_frame(con, **kwargs))
    robustness = pd.concat({n: fits[n].coef for n in ["all stores", *ROBUSTNESS]}, axis=1)
    excluded = con.execute(
        "SELECT split, count(*) AS rows, avg(sale_amount) AS mean_sales FROM daily"
        " WHERE discount = 0 GROUP BY split ORDER BY split DESC"
    ).df()
    return {
        "effects": effects(fits["all stores"]),
        "stability": stability(fits["half A"], fits["half B"]).reset_index(),
        "robustness": robustness.reset_index(),
        "excluded": excluded,
        "timing": pd.DataFrame(timing, columns=["fit", "rows", "seconds"]),
    }


def render_report(results: dict[str, pd.DataFrame]) -> str:
    eff = results["effects"].set_index("term")
    stab = results["stability"]

    def line(term: str) -> str:
        row = eff.loc[term]
        return (
            f"- **{row.meaning}:** {row.effect_pct:+.1f}% sales "
            f"(95% CI {row.ci_low_pct:+.1f}% to {row.ci_high_pct:+.1f}%)."
        )

    parts = [
        "# What drives daily demand",
        "Poisson regression (PPML) of daily sales on the train split (90 days), with store-product "
        "and week fixed effects and standard errors clustered by store. Generated by "
        "`uv run python -m fresh_replenish.analysis.drivers`.",
        "**These are associations, not causes.** Discounts and promotions are not assigned at "
        "random, and sales on stockout days understate demand (Phase 3 recovers it).",
        "## Headline",
        *(line(t) for t in ("log_discount", "activity_flag", "holiday_flag")),
        f"- **Stability:** {int(stab.stable.sum())} of {len(stab)} drivers keep their sign and "
        "size across two random halves of the stores.",
        "## Effects",
        md_table(results["effects"]),
        "## Stability across two random halves of the stores",
        md_table(stab),
        "## Robustness (coefficients)",
        md_table(results["robustness"]),
        "## Excluded rows (discount = 0)",
        md_table(results["excluded"]),
        "## Fit time",
        md_table(results["timing"]),
    ]
    return "\n\n".join(parts) + "\n"


def main(raw_dir: Path = RAW_DIR, out_dir: Path = REPORTS_DIR) -> int:
    paths = [raw_dir / name for name in DATASET_FILES]
    if not all(p.exists() for p in paths):
        print(
            "Data not found. Run: uv run python -m fresh_replenish.data.download",
            file=sys.stderr,
        )
        return 2
    results = run(connect(*paths))
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / "tables" / f"drivers_{name}.csv", index=False)
    report = out_dir / "demand_drivers.md"
    report.write_text(render_report(results), encoding="utf-8")
    print(f"Report written to {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
