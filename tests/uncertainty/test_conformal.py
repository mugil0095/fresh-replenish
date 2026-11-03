import numpy as np
import pandas as pd
import pytest

from fresh_replenish.metrics import DECILE_COLUMNS, DECILES
from fresh_replenish.uncertainty.conformal import POOLED, calibrate, group_table, shifts

CAL = [f"cal_{c}" for c in DECILE_COLUMNS]


def _frame(actual, deciles, model="m"):
    """Forecast rows for series 0..n-1 of store 1: one row per actual."""
    n = len(actual)
    frame = pd.DataFrame({"model": model, "store_id": 1, "product_id": np.arange(n),
                          "actual": actual})
    frame[DECILE_COLUMNS] = np.broadcast_to(deciles, (n, 9))
    return frame


def _groups(n, category=0):
    return pd.DataFrame({"store_id": 1, "product_id": np.arange(n), "group": category})


def test_shift_is_the_conformal_rank_of_spread_scaled_errors():
    calibration = _frame(np.arange(1.0, 10.0), np.linspace(0, 1, 9))  # spread q90 - q10 = 1
    table = shifts(calibration, _groups(9)).set_index("model")
    # n = 9: decile tau takes the ceil(10 * tau)-th smallest of actual - q_tau
    assert table.loc["m", "q10"] == 1.0
    assert table.loc["m", "q50"] == 4.5
    assert table.loc["m", "q90"] == 8.0


def test_calibration_fixes_too_narrow_ranges_on_a_fresh_draw():
    rng = np.random.default_rng(0)
    truth = np.quantile(rng.gamma(4, 0.25, 200_000), DECILES)  # true deciles at level 1
    narrow = truth[4] + 0.5 * (truth - truth[4])  # half as wide as it should be

    def draw(seed, n=6_000):
        r = np.random.default_rng(seed)
        level = r.uniform(1, 10, n)
        frame = _frame(level * r.gamma(4, 0.25, n), 0)
        frame[DECILE_COLUMNS] = level[:, None] * narrow
        return frame

    def coverage(frame, cols):
        return frame.actual.between(frame[cols[0]], frame[cols[-1]]).mean()

    test = calibrate(draw(2), shifts(draw(1), _groups(6_000)), _groups(6_000))
    assert coverage(test, DECILE_COLUMNS) < 0.6
    assert coverage(test, CAL) == pytest.approx(0.8, abs=0.02)


def test_small_categories_share_the_pooled_group():
    ids = pd.DataFrame({"store_id": 1, "product_id": np.arange(15),
                        "first_category_id": [7] * 12 + [8] * 3})
    group = group_table(ids, min_series=10).set_index("product_id").group
    assert set(group[:12]) == {7} and set(group[12:]) == {POOLED}


def test_calibrated_deciles_are_sorted_and_non_negative_and_point_forecasts_kept():
    calibration = _frame(np.full(20, 0.1), np.linspace(1, 2, 9))  # forecasts far too high
    point = _frame(np.ones(20), np.nan, model="seasonal naive")
    point["q50"] = 1.0
    out = calibrate(pd.concat([calibration, point]), shifts(calibration, _groups(20)), _groups(20))
    model = out[out.model == "m"][CAL].to_numpy()
    assert (model >= 0).all() and (np.diff(model, axis=1) >= 0).all()
    assert out[out.model == "seasonal naive"][CAL].isna().all().all()
    assert len(out) == 40
