"""What drives daily demand (spec step 2): Poisson fixed-effects regression (PPML)."""

import duckdb
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
