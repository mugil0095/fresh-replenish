"""What drives daily demand (spec step 2): Poisson fixed-effects regression (PPML)."""

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

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
       weekofyear(CAST(dt AS DATE)) AS week
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
