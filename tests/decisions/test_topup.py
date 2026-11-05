import numpy as np
import pandas as pd
import pytest

from fresh_replenish.config import REPORTS_DIR
from fresh_replenish.decisions import topup
from fresh_replenish.decisions.newsvendor import simulate
from fresh_replenish.decisions.topup import hourly_demand, serve, still_to_come, top_up
from fresh_replenish.forecasting.data import CLASSICAL
from fresh_replenish.imputation.data import Hours
from fresh_replenish.metrics import CALIBRATED_COLUMNS, DECILE_COLUMNS, DECILES

KEYS = ["store_id", "product_id"]


def test_one_delivery_costs_the_same_as_the_daily_simulation():
    rng = np.random.default_rng(0)
    order, early, late = rng.uniform(0, 10, (3, 50, 7))
    waste, lost = serve(order, 0, early, late)
    cost = simulate(order, early + late, r=4)[0]
    assert (waste + 4 * lost).sum(axis=1) == pytest.approx(cost)


def test_serve_loses_what_the_top_up_arrives_too_late_for():
    waste, lost = serve(np.array(5.0), np.array(4.0), np.array(8.0), np.array(3.0))
    assert (waste, lost) == (1.0, 3.0)  # 3 lost before it arrives; 4 delivered, 3 sold


def test_serve_carries_morning_stock_into_the_afternoon():
    waste, lost = serve(np.array(5.0), np.array(1.0), np.array(2.0), np.array(6.0))
    assert (waste, lost) == (0.0, 2.0)  # 3 left + 1 delivered against 6


@pytest.mark.parametrize(
    ("target", "morning", "seen", "units"),
    [(6.0, 5.0, 3.0, 4.0), (1.0, 5.0, 0.0, 0.0), (6.0, 5.0, 9.0, 6.0)],
    ids=["counts the shelf", "never negative", "sold out means an empty shelf"],
)
def test_top_up_orders_up_to_the_target(target, morning, seen, units):
    assert top_up(np.array(target), np.array(morning), np.array(seen)) == units


def _hours(sales, night=None) -> Hours:
    n, days, _ = sales.shape
    night = np.zeros((n, days)) if night is None else night
    return Hours(
        ids=pd.DataFrame({"store_id": np.arange(n), "product_id": np.zeros(n, dtype=int)}),
        dates=np.array([f"2024-01-{d + 1:02d}" for d in range(days)]),
        sales=sales,
        night=night,
        covariates=np.zeros((n, days, 4)),
        sale_amount=np.nansum(sales, axis=-1) + night,
        stockout_hours=np.isnan(sales).sum(axis=-1),
    )


def test_hourly_demand_fills_stockouts_and_counts_night_sales_last():
    sales = np.tile([2.0, 6.0, 2.0], (1, 3, 1))
    sales[0, 2, 2] = np.nan
    night = np.array([[0.0, 1.0, 0.0]])
    demand = hourly_demand(_hours(sales, night))
    assert demand[0, 1].tolist() == [2.0, 6.0, 3.0]
    assert demand[0, 2].tolist() == pytest.approx([2.0, 6.0, 2.0])


def test_hourly_demand_stops_before_the_given_day():
    sales = np.ones((1, 4, 2))
    assert hourly_demand(_hours(sales), stop=3).shape == (1, 3, 2)


def test_still_to_come_is_the_usual_share_left_after_each_hour():
    past = np.array([[[3.0, 1.0, 0.0], [3.0, 1.0, 0.0]]])
    assert still_to_come(past)[0].tolist() == pytest.approx([0.25, 0.0, 0.0])


def test_a_series_with_no_past_demand_borrows_the_pooled_shape():
    past = np.array([[[1.0, 1.0]], [[0.0, 0.0]]])
    assert still_to_come(past)[1].tolist() == pytest.approx([0.5, 0.0])


PAST, HELD_OUT = np.arange(7, 14), np.arange(14, 21)


def _frame(h, days, calibrated):
    """A flat daily forecast made before the held-out week: the deciles of the past days."""
    demand = h.sale_amount[:, days]
    n, k = demand.shape
    frame = pd.DataFrame({
        "model": CLASSICAL,
        "store_id": np.repeat(h.ids.store_id.to_numpy(), k),
        "product_id": np.repeat(h.ids.product_id.to_numpy(), k),
        "dt": np.tile(h.dates[days], n),
        "actual": demand.ravel(),
        "in_stock": True,
    })
    frame[DECILE_COLUMNS] = np.quantile(h.sale_amount[:, PAST], DECILES)
    if calibrated:
        frame["group"] = 0
        frame[CALIBRATED_COLUMNS] = frame[DECILE_COLUMNS].to_numpy()
    return frame


def _scenario(scale=1.0):
    """Morning demand swings; the 16:00 peak never does, so a midday count settles the day."""
    rng = np.random.default_rng(1)
    sales = np.zeros((40, 21, 16))
    sales[..., 1:4] = rng.uniform(0, 20, (40, 21, 3))
    sales[..., 10] = 5.0
    sales[:, 14:] *= scale
    h = _hours(sales)
    return h, _frame(h, PAST, False), _frame(h, HELD_OUT, True)


def test_a_midday_count_beats_one_delivery_when_the_morning_swings():
    results = topup.run(*_scenario(), ratios=(1.0, 9.0), headline=9.0)
    plans = results["plans"]
    assert len(plans) == 2 * len(topup.DECISION_HOURS)
    best = plans[plans.chosen & (plans.ratio == 9.0)].iloc[0]
    assert 10 <= best.decision_hour <= 14
    assert best.arrival_hour == best.decision_hour + topup.LEAD
    assert best.saving_low > 0.1
    assert best.cost < best.one_delivery_cost
    assert best.fill_rate > best.one_delivery_fill_rate
    assert plans.groupby("ratio").chosen.sum().tolist() == [1, 1]


def test_the_held_out_week_plays_no_part_in_choosing_the_plan():
    columns = ["ratio", "decision_hour", "morning_share", "chosen"]
    usual = topup.run(*_scenario(), ratios=(9.0,), headline=9.0)["plans"]
    busier = topup.run(*_scenario(scale=3.0), ratios=(9.0,), headline=9.0)["plans"]
    assert usual[columns].equals(busier[columns])
    assert not np.allclose(usual.cost, busier.cost)


def test_days_with_no_stockout_are_scored_on_their_own():
    h, past, held_out = _scenario()
    held_out["in_stock"] = held_out.dt == held_out.dt.min()
    plans = topup.run(h, past, held_out, ratios=(9.0,), headline=9.0)["plans"]
    every_day = topup.run(*_scenario(), ratios=(9.0,), headline=9.0)["plans"]
    assert not np.allclose(plans.saving_no_stockout, every_day.saving_no_stockout)
    assert np.allclose(plans.saving, every_day.saving)


def test_forecasts_must_cover_every_series():
    h, past, held_out = _scenario()
    with pytest.raises(ValueError, match="every series"):
        topup.run(h, past[past.store_id > 0], held_out, ratios=(9.0,), headline=9.0)


def test_headline_must_be_one_of_the_ratios():
    with pytest.raises(ValueError, match="headline"):
        topup.run(*_scenario(), ratios=(1.0,), headline=9.0)


def test_report_states_the_saving_and_what_it_leaves_out():
    results = topup.run(*_scenario(), ratios=(1.0, 9.0), headline=9.0)
    best = results["plans"].query("chosen and ratio == 9").iloc[0]
    report = topup.render_report(results)
    assert f"{best.saving:.0%} less than one delivery" in report
    assert "does not count what the second delivery itself costs" in report
    assert f"{int(best.decision_hour)}:00" in report


def test_the_saved_one_delivery_costs_match_the_orders_report():
    plans = pd.read_csv(REPORTS_DIR / "tables" / "topup_plans.csv")
    orders = pd.read_csv(REPORTS_DIR / "tables" / "orders_policies.csv")
    orders = orders[orders.policy == CLASSICAL].set_index("ratio").cost
    saved = plans.groupby("ratio").one_delivery_cost.first()
    assert saved.to_numpy() == pytest.approx(orders[saved.index].to_numpy())


def test_main_reports_missing_forecasts(tmp_path, capsys):
    assert topup.main(out_dir=tmp_path, processed_dir=tmp_path) == 2
    assert "forecast_folds.parquet not found" in capsys.readouterr().err
