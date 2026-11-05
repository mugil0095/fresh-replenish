import numpy as np
import pandas as pd
import pytest

from fresh_replenish.config import REPORTS_DIR
from fresh_replenish.decisions import topup
from fresh_replenish.decisions.newsvendor import simulate, top_up
from fresh_replenish.decisions.topup import hourly_demand, serve, still_to_come
from fresh_replenish.forecasting.data import BASELINE, CLASSICAL
from fresh_replenish.imputation.data import Hours
from fresh_replenish.metrics import CALIBRATED_COLUMNS, DECILE_COLUMNS, DECILES
from fresh_replenish.serving import batch

KEYS = ["store_id", "product_id"]


def _days(*values):
    return np.array([values], dtype=float)


def test_one_delivery_costs_the_same_as_the_daily_simulation():
    rng = np.random.default_rng(0)
    order, demand = rng.uniform(0, 10, (2, 50, 7))
    none = np.zeros_like(demand)
    _, _, waste, lost = serve(order, none, none, demand, demand)
    assert (waste + 4 * lost).sum(axis=1) == pytest.approx(simulate(order, demand, r=4)[0])


def test_serve_loses_what_the_top_up_arrives_too_late_for():
    _, extra, waste, lost = serve(_days(5), _days(6), _days(3), _days(8), _days(11))
    assert (extra.item(), waste.item(), lost.item()) == (4.0, 1.0, 3.0)


def test_serve_carries_morning_stock_into_the_afternoon():
    _, extra, waste, lost = serve(_days(5), _days(4), _days(2), _days(2), _days(8))
    assert (extra.item(), waste.item(), lost.item()) == (1.0, 0.0, 2.0)


def test_the_oldest_stock_sells_first():
    stock = np.array([[2.0, 3.0]])
    unmet = topup._sell(stock, np.array([3.0]))
    assert stock.tolist() == [[0.0, 2.0]] and unmet.tolist() == [0.0]


def test_demand_beyond_the_stock_goes_unmet():
    assert topup._sell(np.array([[2.0, 3.0]]), np.array([7.0])).tolist() == [2.0]


def test_unsold_stock_sells_the_next_day_when_it_keeps():
    up_to, demand, none = _days(5, 5), _days(3, 3), _days(0, 0)
    assert serve(up_to, none, none, demand, demand)[2].tolist() == [[2.0, 2.0]]
    morning, _, waste, lost = serve(up_to, none, none, demand, demand, life=2)
    assert morning.tolist() == [[3.0, 3.0]] and not waste.any() and not lost.any()


def test_stock_is_thrown_away_when_its_life_ends():
    none = _days(0, 0)
    morning, _, waste, _ = serve(_days(5, 5), none, none, none, none, life=2)
    assert morning.tolist() == [[5.0, 0.0]] and waste.tolist() == [[0.0, 5.0]]


@pytest.mark.parametrize(
    ("target", "shelf", "units"),
    [(6.0, 2.0, 4.0), (1.0, 5.0, 0.0)],
    ids=["counts the shelf", "never negative"],
)
def test_top_up_orders_up_to_the_target(target, shelf, units):
    assert top_up(np.array(target), np.array(shelf)) == units


def _one_day():
    """Hours 6:00 to 10:00 sell 1, 2, 4, 8 and 16, so any slice's total names its hours."""
    by_hour = np.cumsum([1.0, 2, 4, 8, 16] + [0.0] * 11)
    frame = pd.DataFrame({"model": [CLASSICAL], "store_id": [0], "product_id": [0]})
    return topup.Week(
        frame, np.ones((1, 1, 9)), np.array([[31.0]]), by_hour[None, None],
        np.full((1, 16), 0.5), np.array([[True]]),
    )


def test_a_plan_sees_the_hours_before_it_decides_and_serves_those_before_it_lands():
    target, order = np.array([[20.0]]), np.array([[10.0]])
    delivered, extra, waste, lost = topup._plan(_one_day(), order, target, 8, 0.5)
    assert extra[0, 0] == 18.0  # 3 sold by 8:00, so 2 of the 5 are left against a target of 20
    assert lost[0, 0] == 10.0  # 15 wanted before the 10:00 delivery, 5 on hand
    assert waste[0, 0] == 2.0  # 18 delivered against the last 16
    assert delivered[0, 0] == 23.0


def test_a_morning_that_sold_out_leaves_an_empty_shelf():
    _, extra, _, _ = topup._plan(_one_day(), np.array([[4.0]]), np.array([[20.0]]), 8, 0.5)
    assert extra[0, 0] == 20.0  # 2 delivered, 3 wanted by 8:00


def test_the_rest_of_the_day_starts_at_the_decision_hour():
    rest = topup._rest_of_day(_one_day(), 8)
    assert rest.actual.tolist() == [28.0]  # 31 less the 3 sold before 8:00
    assert rest[DECILE_COLUMNS].to_numpy().tolist() == [[0.5] * 9]


def test_every_top_up_lands_before_closing():
    assert max(topup.DECISION_HOURS) + topup.LEAD < topup.CLOSING


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


def _served(results, h, held_out):
    """The held-out week's orders as the batch would write them for the plan `run` chose."""
    orders = held_out[[*KEYS, "dt"]].copy()
    orders[DECILE_COLUMNS] = held_out[CALIBRATED_COLUMNS].to_numpy()
    plan = results["shifts"]
    t = plan.decision_hour.iloc[0] - topup.OPENING
    first = int(np.isin(h.dates, held_out.dt.unique()).argmax())
    left = h.ids.assign(left=still_to_come(hourly_demand(h, first))[:, t - 1])
    ids = h.ids.assign(first_category_id=0)
    return batch.with_top_up(orders, left, ids, plan)


def _demand(held_out):
    return held_out[[*KEYS, "dt"]].assign(
        recovered_demand=held_out.actual, stockout_hours=np.where(held_out.in_stock, 0.0, 1.0)
    )


def test_run_hands_the_batch_the_plan_it_chose():
    results = topup.run(*_scenario(), ratios=(1.0, 9.0), headline=9.0)
    plans, shifts = results["plans"], results["shifts"]
    best = plans[plans.chosen & (plans.ratio == 9.0)].iloc[0]
    assert (shifts.decision_hour == best.decision_hour).all()
    assert (shifts.arrival_hour == best.arrival_hour).all()
    assert (shifts.morning_share == best.morning_share).all()
    assert shifts[DECILE_COLUMNS].notna().all(axis=None)


def test_the_served_plan_replays_the_study_exactly():
    h, past, held_out = _scenario()
    results = topup.run(h, past, held_out, ratios=(1.0, 9.0), headline=9.0)
    best = results["plans"].query("chosen and ratio == 9").iloc[0]
    row = topup.replay(h, _served(results, h, held_out), _demand(held_out)).iloc[0]
    assert row.cost == pytest.approx(best.cost)
    assert row.one_delivery_cost == pytest.approx(best.one_delivery_cost)
    assert row.saving_low == pytest.approx(best.saving_low)
    assert (row.series, row.stores) == (40, 40)


def test_replay_scores_only_the_hours_it_was_served():
    h, past, held_out = _scenario()
    results = topup.run(h, past, held_out, ratios=(9.0,), headline=9.0)
    served = _served(results, h, held_out)
    row = topup.replay(h, served, _demand(held_out)).iloc[0]
    assert row.decision_hour == served.decision_hour.iloc[0]
    assert row.morning_share == served.morning_share.iloc[0]


def test_forecasts_must_cover_every_series():
    h, past, held_out = _scenario()
    with pytest.raises(ValueError, match="every series"):
        topup.run(h, past[past.store_id > 0], held_out, ratios=(9.0,), headline=9.0)


def test_headline_must_be_one_of_the_ratios():
    with pytest.raises(ValueError, match="headline"):
        topup.run(*_scenario(), ratios=(1.0,), headline=9.0)


def _forecasts(held_out):
    return pd.concat([held_out, held_out.assign(model=BASELINE)])


def test_past_weeks_retune_one_delivery_when_stock_keeps():
    short = np.array([[10.0]])  # 31 are wanted: at 9x the highest level is the cheapest
    assert topup._best_level(_one_day(), short, 9.0, 2) == max(topup.MORNING_SHARES)
    plenty = np.array([[62.0]])
    assert topup._best_level(_one_day(), plenty, 9.0, 2) == 0.5


def test_one_day_shelf_life_keeps_the_newsvendor_level():
    plans = topup.run(*_scenario(), ratios=(9.0,), headline=9.0)["plans"]
    assert (plans.one_delivery_level == 1).all()


def test_the_top_up_is_judged_against_the_retuned_delivery():
    h, past, held_out = _scenario()
    row = topup.shelf_life(h, past, _forecasts(held_out), r=9.0, lives=(2,)).iloc[0]
    assert row.retuned_level != 1
    assert row.top_up_saving == pytest.approx(1 - row.top_up_cost / row.retuned_cost)


def test_shelf_life_starts_from_the_one_day_results():
    h, past, held_out = _scenario()
    table = topup.shelf_life(h, past, _forecasts(held_out), r=9.0, lives=(1, 2))
    plans = topup.run(h, past, held_out, ratios=(9.0,), headline=9.0)["plans"]
    best, first = plans[plans.chosen].iloc[0], table.iloc[0]
    assert table.life_days.tolist() == [1, 2]
    assert first.cost_based_cost == pytest.approx(best.one_delivery_cost)
    assert first.top_up_cost == pytest.approx(best.cost)
    assert first.top_up_saving == pytest.approx(best.saving)
    assert first.top_up_saving_low == pytest.approx(best.saving_low)
    assert first.retuned_cost == first.cost_based_cost and first.retuned_level == 1
    assert table.cost_based_cost.iloc[1] < first.cost_based_cost


def test_shelf_life_compares_each_policy_on_the_same_resamples():
    h, past, held_out = _scenario()
    row = topup.shelf_life(h, past, _forecasts(held_out), r=9.0, lives=(1,)).iloc[0]
    assert row.saving_vs_median == pytest.approx(1 - row.cost_based_cost / row.median_cost)
    assert row.saving_vs_median_low <= row.saving_vs_median <= row.saving_vs_median_high
    assert row.saving_vs_last_week == pytest.approx(1 - row.cost_based_cost / row.last_week_cost)


def test_the_saved_one_day_shelf_life_matches_the_orders_report():
    table = pd.read_csv(REPORTS_DIR / "tables" / "topup_shelf_life.csv")
    orders = pd.read_csv(REPORTS_DIR / "tables" / "orders_policies.csv")
    orders = orders[orders.ratio == 9].set_index("policy").cost
    first = table.iloc[0]
    assert first.life_days == 1
    assert first.cost_based_cost == pytest.approx(orders[CLASSICAL])
    assert first.median_cost == pytest.approx(orders[f"{CLASSICAL} median"])
    assert first.last_week_cost == pytest.approx(orders[BASELINE])


def test_report_states_the_saving_and_what_it_leaves_out():
    h, past, held_out = _scenario()
    results = topup.run(h, past, held_out, ratios=(1.0, 9.0), headline=9.0)
    results["shelf_life"] = topup.shelf_life(h, past, _forecasts(held_out), r=9.0, lives=(1, 2))
    best = results["plans"].query("chosen and ratio == 9").iloc[0]
    report = topup.render_report(results)
    assert "## If stock keeps longer" in report
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
