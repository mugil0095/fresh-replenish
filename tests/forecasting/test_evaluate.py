import shutil

import numpy as np
import pytest

from fresh_replenish.forecasting import evaluate
from fresh_replenish.forecasting.models import seasonal_naive


def _perfect(series, origin, horizon):
    return np.repeat(series.demand[:, None, origin : origin + horizon], 9, axis=1)


def _biased(series, origin, horizon):
    return _perfect(series, origin, horizon) * 1.5


FAKES = {"seasonal naive": seasonal_naive, "perfect": _perfect, "lightgbm": _biased}


@pytest.fixture
def results(make_series, weekly):
    return evaluate.run(make_series(weekly(n=12, days=42)), FAKES, origins=(28, 35))


def test_methods_table(results):
    methods = results[0]["methods"].set_index("model")
    assert list(methods.index) == list(FAKES)
    assert methods.loc["perfect", "wape"] == 0
    assert methods.loc["lightgbm", "wpe"] == pytest.approx(0.5)
    assert methods.loc["perfect", "coverage_80"] == 1.0
    assert np.isnan(methods.loc["seasonal naive", "wql"])  # a point forecast has no deciles


def test_paired_differences_against_best_and_lightgbm(results):
    perfect = results[0]["methods"].set_index("model").loc["perfect"]
    assert perfect[["vs_best", "vs_best_low", "vs_best_high"]].tolist() == [0, 0, 0]
    assert perfect[["vs_lightgbm", "vs_lightgbm_low", "vs_lightgbm_high"]].tolist() == (
        pytest.approx([-0.5] * 3)
    )


def test_folds_flag_models_that_beat_seasonal_naive_every_fold(results):
    folds = results[0]["folds"].set_index("model")
    assert list(folds.columns[:2]) == ["fold 1", "fold 2"]
    assert folds.loc["perfect", "beats_seasonal_naive"] is True
    assert folds.loc["lightgbm", "beats_seasonal_naive"] is False


def test_fold_forecasts_have_a_row_per_model_series_and_day(results):
    saved = results[1]
    assert len(saved) == len(FAKES) * 12 * 2 * 7
    assert list(saved.columns[:7]) == [
        "model", "fold", "store_id", "product_id", "dt", "actual", "in_stock"
    ]
    perfect = saved[saved.model == "perfect"]
    assert (perfect.q50 == perfect.actual).all()
    assert set(perfect.fold) == {1, 2}


def test_report_names_the_best_model_and_the_misses(results):
    report = evaluate.render_report(results[0])
    assert "**Best model: perfect.** WAPE 0.0%" in report
    assert "do not: lightgbm" in report


def test_main_without_data_returns_2(tmp_path):
    assert evaluate.main(raw_dir=tmp_path / "empty", out_dir=tmp_path, processed_dir=tmp_path) == 2


def test_main_without_recovered_demand_returns_2(tiny_dataset, tmp_path, capsys):
    raw = tmp_path / "raw" / "data"
    raw.mkdir(parents=True)
    for path in tiny_dataset:
        shutil.copy(path, raw / path.name)
    assert evaluate.main(raw_dir=raw.parent, out_dir=tmp_path, processed_dir=tmp_path) == 2
    assert "fresh_replenish.imputation.recovery" in capsys.readouterr().err
