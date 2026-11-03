import shutil

import numpy as np
import pandas as pd
import pytest

from fresh_replenish.forecasting import evaluate
from fresh_replenish.forecasting.models import seasonal_naive
from fresh_replenish.metrics import DECILE_COLUMNS, DECILES
from fresh_replenish.uncertainty import conformal
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


def _narrow(series, origin, horizon):
    """Last week's demand with a range far too tight to cover 80%."""
    point = seasonal_naive(series, origin, horizon)[:, 4]
    return point[:, None, :] * np.linspace(0.96, 1.04, 9)[None, :, None]


@pytest.fixture
def calibrated(make_series, weekly):
    series = make_series(weekly(n=12, days=49))  # the last 7 days play the held-out week
    fakes = {"seasonal naive": seasonal_naive, "narrow": _narrow}
    folds = evaluate.run(series, fakes, origins=(21, 28, 35))[1]
    return conformal.run(series, folds, fakes)


def test_run_calibrates_the_held_out_week(calibrated):
    results, forecasts = calibrated
    coverage = results["coverage"].set_index("model")
    assert list(coverage.index) == ["narrow"]  # a point forecast has no range to calibrate
    assert coverage.loc["narrow", "coverage_after"] > coverage.loc["narrow", "coverage_before"]
    assert len(forecasts) == 2 * 12 * 7 and set(forecasts.dt) == {str(d) for d in range(42, 49)}
    assert {"q10", "cal_q10", "group"} <= set(forecasts.columns)


def test_report_states_the_exit_check(calibrated):
    report = conformal.render_report(calibrated[0])
    assert report.startswith("# Calibrated forecast ranges")
    assert "**Exit check:**" in report and "narrow" in report


def test_main_without_data_returns_2(tmp_path):
    assert conformal.main(raw_dir=tmp_path / "none", out_dir=tmp_path, processed_dir=tmp_path) == 2


def test_main_without_fold_forecasts_names_phase_4(tiny_dataset, tmp_path, capsys):
    raw = tmp_path / "raw" / "data"
    raw.mkdir(parents=True)
    for path in tiny_dataset:
        shutil.copy(path, raw / path.name)
    (tmp_path / "recovered_demand.parquet").touch()
    assert conformal.main(raw_dir=raw.parent, out_dir=tmp_path, processed_dir=tmp_path) == 2
    assert "fresh_replenish.forecasting.evaluate" in capsys.readouterr().err


def test_calibrate_refuses_a_group_without_shifts():
    calibration = _frame(np.ones(20), np.linspace(0, 2, 9))
    table = shifts(calibration, _groups(20))
    with pytest.raises(ValueError, match="no calibration shift"):
        calibrate(calibration, table, _groups(20, category=5))
