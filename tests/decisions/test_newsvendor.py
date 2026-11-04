import numpy as np
import pandas as pd
import pytest

from fresh_replenish.decisions import newsvendor
from fresh_replenish.decisions.newsvendor import critical_ratio, order_at, simulate
from fresh_replenish.metrics import DECILE_COLUMNS

CALIBRATED = [f"cal_{c}" for c in DECILE_COLUMNS]


def test_critical_ratio_is_the_cost_share_of_a_lost_sale():
    assert critical_ratio(9) == pytest.approx(0.9)


def test_the_lowest_ratio_never_orders_below_the_lowest_decile():
    q = np.array([[0.0, 5, 6, 7, 8, 9, 10, 11, 12]])  # critical_ratio(1/9) is 0.0999...
    assert order_at(q, critical_ratio(1 / 9))[0] == 0.0
    assert critical_ratio(1) == 0.5


def test_order_interpolates_between_neighbouring_deciles():
    q = np.arange(9.0)  # q10 = 0 ... q90 = 8
    assert order_at(q, 0.85) == pytest.approx(7.5)
    assert order_at(q, 0.9) == pytest.approx(8.0)
    assert order_at(q, 0.1) == pytest.approx(0.0)
    assert order_at(q, 1 / 3) == pytest.approx(7 / 3)


@pytest.mark.parametrize("ratio", [0.05, 0.95])
def test_order_outside_the_deciles_raises(ratio):
    with pytest.raises(ValueError, match="outside"):
        order_at(np.arange(9.0), ratio)


def test_simulate_charges_waste_and_lost_sales():
    cost, bias, demand = simulate(np.array([[3.0, 5.0]]), np.array([[5.0, 3.0]]), r=2)
    assert cost.tolist() == [6.0]  # 2 lost x 2 + 2 wasted x 1
    assert bias.tolist() == [0.0]
    assert demand.tolist() == [8.0]


def _forecasts():
    """Held-out forecasts for 2 stores x 3 products x 7 days from three models."""
    keys = pd.DataFrame(
        [(s, p, str(d)) for s in (1, 2) for p in (10, 11, 12) for d in range(7)],
        columns=["store_id", "product_id", "dt"],
    )
    rng = np.random.default_rng(0)
    actual = rng.uniform(1, 5, len(keys))
    guess = actual * rng.uniform(0.9, 1.1, len(keys))  # a median that is off by up to 10%
    around = guess[:, None] * np.linspace(0.8, 1.2, 9)
    exact = np.repeat(actual[:, None], 9, axis=1)
    frames = []
    for model, deciles in (("seasonal naive", None), ("lightgbm", around), ("oracle", exact)):
        frame = keys.assign(model=model, actual=actual)
        frame[DECILE_COLUMNS] = np.nan if deciles is None else deciles
        frame[CALIBRATED] = np.nan if deciles is None else deciles
        frames.append(frame)
    frames[0]["q50"] = actual * 1.2  # seasonal naive: a point forecast, 20% high
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def results():
    return newsvendor.run(_forecasts())


def test_an_exact_forecast_costs_nothing_and_is_cheapest(results):
    p = results["policies"]
    assert set(p.policy) == {"seasonal naive", "oracle median", "lightgbm", "oracle"}
    assert len(p) == len(newsvendor.RATIOS) * 4
    oracle = p[p.policy == "oracle"]
    assert np.allclose(oracle.cost, 0) and np.allclose(oracle.vs_cheapest, 0)


def test_paired_differences_against_the_lowest_error_model(results):
    p = results["policies"]
    assert results["sample"].lowest_error_model.iloc[0] == "oracle"
    assert np.allclose(p[p.policy == "oracle"].vs_lowest_error, 0)
    assert (p[p.policy == "lightgbm"].vs_lowest_error_low > 0).all()
    assert (p[p.policy == "seasonal naive"].saving_vs_naive == 0).all()


def test_a_dearer_lost_sale_orders_more(results):
    p = results["policies"]
    fill = p[p.policy == "lightgbm"].sort_values("ratio").fill_rate
    assert fill.is_monotonic_increasing and fill.iloc[-1] == 1.0


def test_store_counts_cover_every_store(results):
    stores = results["stores"].set_index("policy").stores
    assert stores.sum() == 2 and stores["oracle"] + stores["oracle median"] == 2  # both exact


def test_report_names_the_cheapest_policy(results):
    report = newsvendor.render_report(results)
    assert report.startswith("# Cost-based orders")
    assert "**Cheapest at r = 9:** oracle" in report


def test_a_model_missing_rows_is_refused():
    forecasts = _forecasts()
    with pytest.raises(ValueError, match="every series and day"):
        newsvendor.run(forecasts.drop(index=forecasts.index[forecasts.model == "lightgbm"][0]))


def test_headline_must_be_one_of_the_ratios():
    with pytest.raises(ValueError, match="headline"):
        newsvendor.run(_forecasts(), ratios=(1.0, 2.0))


def test_main_without_held_out_forecasts_names_the_calibration_step(tmp_path, capsys):
    assert newsvendor.main(out_dir=tmp_path, processed_dir=tmp_path) == 2
    assert "fresh_replenish.uncertainty.conformal" in capsys.readouterr().err
