import numpy as np
import pandas as pd
import pytest

from fresh_replenish.decisions import weekly
from fresh_replenish.metrics import CALIBRATED_COLUMNS, DECILE_COLUMNS


def _folds(scale_first=1.0):
    """3 past weeks of forecasts for 2 stores x 3 products; LightGBM's median is up to 10% off."""
    frames, rng = [], np.random.default_rng(0)
    for fold in (1, 2, 3):
        keys = pd.DataFrame(
            [(s, p, f"2024-05-{7 * fold + d:02d}") for s in (1, 2) for p in (10, 11, 12)
             for d in range(7)],
            columns=["store_id", "product_id", "dt"],
        )
        actual = rng.uniform(1, 5, len(keys)) * (scale_first if fold == 1 else 1.0)
        guess = actual / (scale_first if fold == 1 else 1.0) * rng.uniform(0.9, 1.1, len(keys))
        for model, deciles in (("seasonal naive", None),
                               ("lightgbm", guess[:, None] * np.linspace(0.6, 1.4, 9))):
            frame = keys.assign(model=model, fold=fold, actual=actual, in_stock=True)
            frame[DECILE_COLUMNS] = np.nan if deciles is None else deciles
            if deciles is None:
                frame["q50"] = actual * 1.2  # reorder last week: a point forecast
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


GROUPS = pd.DataFrame({"store_id": [1, 1, 1, 2, 2, 2], "product_id": [10, 11, 12] * 2,
                       "group": 0})


def test_each_week_is_calibrated_on_the_other_weeks_only():
    usual = dict(weekly.calibrated_weeks(_folds(), GROUPS))
    busier = dict(weekly.calibrated_weeks(_folds(scale_first=3.0), GROUPS))
    lightgbm = lambda frame: frame[frame.model == "lightgbm"][CALIBRATED_COLUMNS].to_numpy()  # noqa: E731
    assert np.allclose(lightgbm(usual[1]), lightgbm(busier[1]))  # its own demand plays no part
    assert not np.allclose(lightgbm(usual[2]), lightgbm(busier[2]))  # week 1 calibrates week 2


def test_one_row_per_week_with_the_headline_savings():
    table = weekly.weekly(_folds(), GROUPS)
    assert table.fold.tolist() == [1, 2, 3] and (table.model == "lightgbm").all()
    expected = 1 - table.cost_based_cost / table.median_cost
    assert table.saving_vs_median.to_numpy() == pytest.approx(expected)
    assert table.saving_vs_last_week.to_numpy() == pytest.approx(
        1 - table.cost_based_cost / table.last_week_cost
    )
    assert (table.saving_vs_median > 0).all()  # at 9x the 90% level beats ordering the median


def test_the_report_puts_the_weeks_next_to_the_held_out_one():
    table = weekly.weekly(_folds(), GROUPS)
    held = {"saving_vs_median": 0.56, "saving_vs_last_week": 0.55}
    report = weekly.render_report(table, held)
    assert "56% on the held-out week" in report and "## Every week" in report


def test_main_names_the_missing_step(tmp_path, capsys):
    assert weekly.main(out_dir=tmp_path, processed_dir=tmp_path) == 2
    assert "forecast_folds.parquet not found" in capsys.readouterr().err
