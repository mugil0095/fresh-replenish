import math

import numpy as np
import pandas as pd
import pytest

from fresh_replenish.analysis.drivers import (
    TERMS,
    build_frame,
    effects,
    fit_drivers,
    render_report,
    split_stores,
    stability,
)
from fresh_replenish.data.db import connect

TRUE = {
    "log_discount": -1.5,
    "activity_flag": 0.3,
    "holiday_flag": 0.2,
    "precpt": -0.01,
    "avg_temperature": 0.02,
    "avg_humidity": -0.005,
    "avg_wind_level": -0.05,
}


def simulate(n_series: int = 300, days: int = 28, seed: int = 1) -> pd.DataFrame:
    """Poisson sales with known effects, scaled by 0.1 like the normalised real data."""
    rng = np.random.default_rng(seed)
    n = n_series * days
    df = pd.DataFrame(
        {
            "store_id": np.repeat(np.arange(n_series) % 30, days),
            "product_id": np.repeat(np.arange(n_series) // 30, days),
            "city_id": 0,
            "week": np.tile(np.arange(days) // 7 + 1, n_series),
            "log_discount": np.log(rng.uniform(0.5, 1.0, n)),
            "activity_flag": rng.integers(0, 2, n),
            "holiday_flag": np.tile((np.arange(days) % 7 >= 5).astype(int), n_series),
            "precpt": rng.exponential(5, n),
            "avg_temperature": rng.normal(25, 4, n),
            "avg_humidity": rng.normal(75, 8, n),
            "avg_wind_level": rng.normal(2, 0.5, n),
        }
    )
    eta = np.repeat(rng.normal(0, 0.5, n_series), days) + sum(TRUE[t] * df[t] for t in TERMS)
    df["sale_amount"] = rng.poisson(np.exp(eta)) * 0.1
    df.loc[:days - 1, "sale_amount"] = 0.0  # series 0 never sells (PPML separation)
    return df


def frame(paths, **kwargs):
    return build_frame(connect(*paths), **kwargs)


def test_frame_is_train_only_with_terms(tiny_dataset):
    f = frame(tiny_dataset)
    assert len(f) == 12
    assert set(TERMS) | {"sale_amount", "store_id", "product_id", "city_id", "week"} <= set(f)


def test_zero_discount_rows_are_excluded(make_dataset):
    def zero(train, eval_):
        train[3]["discount"] = 0.0

    assert len(frame(make_dataset(zero))) == 11


def test_log_discount_is_log_price_ratio(make_dataset):
    def half_price(train, eval_):
        train[2]["discount"] = 0.5

    values = sorted(frame(make_dataset(half_price)).log_discount.round(6).unique())
    assert values == [round(math.log(0.5), 6), 0.0]


def test_in_stock_only_drops_stockout_days(tiny_dataset):
    assert len(frame(tiny_dataset, in_stock_only=True)) == 11  # fixture row 0 has 2 stockout hours


def test_exclude_cities(tiny_dataset):
    f = frame(tiny_dataset, exclude_cities=(0,))
    assert set(f.city_id) == {1}
    assert len(f) == 6


def test_week_number(tiny_dataset):
    assert set(frame(tiny_dataset).week) == {"2024-01-01"}  # Monday that starts the week


def test_week_key_is_unique_across_years(make_dataset):
    def next_year(train, eval_):
        train[0]["dt"] = "2024-12-30"  # ISO week 1 of 2025, like 2024-01-01

    assert "2024-12-30" in set(frame(make_dataset(next_year)).week)


def test_fit_recovers_known_effects():
    fit = fit_drivers(simulate())
    assert list(fit.index) == TERMS
    for term, true in TRUE.items():
        assert abs(fit.loc[term, "coef"] - true) < 4 * fit.loc[term, "se"], term


def test_split_stores_partitions():
    df = simulate(n_series=60, days=7)
    a, b = split_stores(df, seed=42)
    assert not set(a.store_id) & set(b.store_id)
    assert len(a) + len(b) == len(df)
    assert split_stores(df, seed=42)[0].equals(a)


def _fit(coefs: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"coef": [c for c, _ in coefs.values()], "se": [s for _, s in coefs.values()]},
        index=pd.Index(list(coefs), name="term"),
    )


def test_stability_flags_planted_difference():
    a = _fit({"activity_flag": (0.30, 0.05), "log_discount": (-1.5, 0.1)})
    b = _fit({"activity_flag": (0.31, 0.05), "log_discount": (-0.5, 0.1)})
    table = stability(a, b)
    assert bool(table.loc["activity_flag", "stable"])
    assert not bool(table.loc["log_discount", "stable"])


def test_stability_missing_term_is_unstable():
    a = _fit({"activity_flag": (0.30, 0.05), "holiday_flag": (0.2, 0.05)})
    b = _fit({"activity_flag": (0.30, 0.05)})  # pyfixest dropped holiday_flag in this half
    assert not bool(stability(a, b).loc["holiday_flag", "stable"])


def _full_fit(coefs: dict[str, float], se: float = 0.01) -> pd.DataFrame:
    fit = _fit({t: (c, se) for t, c in coefs.items()})
    fit["p"] = 0.001
    return fit


def test_effects_price_elasticity():
    row = effects(_full_fit({"log_discount": -1.5})).set_index("term").loc["log_discount"]
    assert row.effect_pct == pytest.approx((0.9**-1.5 - 1) * 100)  # a 10% price cut: +17.1%
    assert row.ci_low_pct < row.effect_pct < row.ci_high_pct


def test_effects_percent_change():
    table = effects(_full_fit({"activity_flag": math.log(1.3)})).set_index("term")
    assert table.loc["activity_flag", "effect_pct"] == pytest.approx(30.0)


def test_report_says_associations():
    fit = _full_fit(dict.fromkeys(TERMS, 0.1))
    results = {
        "effects": effects(fit),
        "stability": stability(fit, fit).reset_index(),
        "robustness": fit[["coef"]].reset_index(),
        "excluded": pd.DataFrame({"split": ["train"], "rows": [10], "mean_sales": [15.0]}),
        "timing": pd.DataFrame({"fit": ["all stores"], "rows": [100], "seconds": [1.0]}),
    }
    report = render_report(results)
    assert "associations, not causes" in report
    assert "7 of 7 drivers" in report
