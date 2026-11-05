"""A second delivery: count the shelf at midday and top up for the rest of the day.

One delivery (`newsvendor.py`) commits the whole day's order before opening. Here the morning
carries a share of that order; at the decision hour the store counts what is left and orders up
to a calibrated target for the rest of the day, delivered LEAD hours later. Days end as before:
unsold units are wasted and missed sales are not made up. `shelf_life` then asks what changes if
unsold units keep for more than one day.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fresh_replenish.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from fresh_replenish.data.db import connect, sample_stores
from fresh_replenish.data.download import local_paths
from fresh_replenish.decisions.newsvendor import (
    HEADLINE,
    RATIOS,
    critical_ratio,
    order_at,
    top_up,
)
from fresh_replenish.forecasting import evaluate
from fresh_replenish.forecasting.data import BASELINE, CLASSICAL
from fresh_replenish.imputation.data import Hours, load_hours
from fresh_replenish.imputation.statistical import profile_recover
from fresh_replenish.metrics import (
    CALIBRATED_COLUMNS,
    DECILE_COLUMNS,
    REST_COLUMNS,
    SEED,
    bootstrap,
)
from fresh_replenish.report import md_table
from fresh_replenish.uncertainty.conformal import KEYS, calibrate, shifts

OPENING, CLOSING = 6, 22  # the hourly arrays cover 6:00 to 22:00
LEAD = 2  # hours from order to delivery: an assumption, the data has no delivery times
DECISION_HOURS = range(8, CLOSING - LEAD - 1)  # the last top-up lands an hour before closing
MORNING_SHARES = np.round(np.arange(4, 21) / 10, 1)  # of the one-delivery order
LIVES = (1, 2, 3)  # days unsold stock keeps: an assumption, the data has no shelf lives


def hourly_demand(h: Hours, stop: int | None = None) -> np.ndarray:
    """Demand per operating hour for the days before `stop`, stockouts filled as in Phase 3.

    Night sales count as the last hour's, so a midday count can never see them.
    """
    hourly = profile_recover(h.sales[:, :stop]).astype(float)
    hourly[..., -1] += h.night[:, :stop]
    return hourly


def still_to_come(past: np.ndarray) -> np.ndarray:
    """Each series' usual share of a day's demand left after each hour, as (series, hours)."""
    usual = past.sum(axis=1)
    total = usual.sum(axis=1, keepdims=True)
    share = np.where(total > 0, usual / np.maximum(total, 1e-9), usual.sum(axis=0) / usual.sum())
    return 1 - share.cumsum(axis=1)


def _sell(stock: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """Sell from `stock` (series, days left), oldest first; returns the demand left unmet."""
    for age in range(stock.shape[1]):
        sold = np.minimum(stock[:, age], demand)
        stock[:, age] -= sold
        demand = demand - sold
    return demand


def serve(up_to, target, seen, early, demand, life: int = 1) -> tuple:
    """Units delivered in the morning, topped up, wasted and lost, each as (series, days).

    Each morning the shelf is filled up to `up_to`. After `seen` demand the store counts it and
    orders up to `target`, which lands after `early` demand. The oldest units sell first, and
    units are thrown away after `life` days. Stock that keeps needs a shelf that isn't empty on
    the first day, so the days run twice and the second pass is what comes back.
    """
    stock = np.zeros((len(demand), life))  # column 0 is thrown away tonight
    morning, extra, waste, lost = (np.zeros(demand.shape) for _ in range(4))
    for d in [*range(demand.shape[1])] * min(life, 2):
        morning[:, d] = np.maximum(up_to[:, d] - stock.sum(axis=1), 0)
        stock[:, -1] += morning[:, d]
        extra[:, d] = top_up(target[:, d], np.maximum(stock.sum(axis=1) - seen[:, d], 0))
        unmet = _sell(stock, early[:, d])
        stock[:, -1] += extra[:, d]
        lost[:, d] = unmet + _sell(stock, demand[:, d] - early[:, d])
        waste[:, d] = stock[:, 0]
        stock = np.column_stack([stock[:, 1:], np.zeros(len(demand))])
    return morning, extra, waste, lost


@dataclass
class Week:
    frame: pd.DataFrame  # one row per series and day, sorted by series, then day
    deciles: np.ndarray  # (series, days, 9): the calibrated daily forecast
    demand: np.ndarray  # (series, days)
    by_hour: np.ndarray  # (series, days, hours): demand so far at the end of each hour
    left: np.ndarray  # (series, hours): usual share of demand left after each hour
    no_stockout: np.ndarray  # (series, days): hourly demand fully observed


def _week(frame: pd.DataFrame, h: Hours) -> Week:
    """Forecast rows as arrays; shares come only from the days before the first forecast day."""
    frame = frame.sort_values([*KEYS, "dt"])
    if not np.array_equal(frame[KEYS].drop_duplicates().to_numpy(), h.ids[KEYS].to_numpy()):
        raise ValueError("forecasts must cover every series in the hourly data, in the same order")
    n, days = len(h.ids), np.isin(h.dates, frame.dt.unique())
    demand = frame.actual.to_numpy(dtype=float).reshape(n, -1)
    hourly = hourly_demand(h)[:, days]
    total = hourly.sum(axis=-1, keepdims=True)
    spread = np.where(total > 0, hourly / np.maximum(total, 1e-9), 1 / hourly.shape[-1])
    return Week(
        frame=frame,
        deciles=frame[CALIBRATED_COLUMNS].to_numpy(dtype=float).reshape(n, demand.shape[1], -1),
        demand=demand,
        by_hour=(spread * demand[..., None]).cumsum(axis=-1),  # ends at the day's demand
        left=still_to_come(hourly_demand(h, int(days.argmax()))),
        no_stockout=frame.in_stock.to_numpy().reshape(demand.shape),
    )


def _rest_of_day(week: Week, hour: int) -> pd.DataFrame:
    """A forecast frame for the demand left at the decision hour: the day's deciles, scaled."""
    t = hour - OPENING
    rest = week.demand - week.by_hour[..., t - 1]
    frame = week.frame[["model", *KEYS]].assign(actual=rest.ravel())
    scaled = week.deciles * week.left[:, None, t - 1, None]
    frame[DECILE_COLUMNS] = scaled.reshape(-1, len(DECILE_COLUMNS))
    return frame


def _plan(week: Week, order, target, hour: int, share: float, life: int = 1) -> tuple:
    """Units delivered, topped up, wasted and lost on every line-day under one plan."""
    t = hour - OPENING
    seen, early = week.by_hour[..., t - 1], week.by_hour[..., t + LEAD - 1]
    morning, extra, waste, lost = serve(share * order, target, seen, early, week.demand, life)
    return morning + extra, extra, waste, lost


def _cost(week: Week, order, target, hour: int, share: float, r: float, life: int) -> float:
    *_, waste, lost = _plan(week, order, target, hour, share, life)
    return (waste + r * lost).sum() / week.demand.sum()


def _draws(cost: np.ndarray, week: Week) -> np.ndarray:
    """Bootstrap draws of cost per unit of demand; the seed is fixed, so policies are paired."""
    sums = cost.sum(axis=1)
    return bootstrap((sums, sums, week.demand.sum(axis=1)))[1]


def _one_delivery(week: Week, order, r: float, life: int = 1) -> tuple:
    """One delivery's waste, lost sales and draws, on all days and on days with no stockout."""
    none = np.zeros_like(week.demand)
    *_, waste, lost = serve(order, none, none, week.demand, week.demand, life)
    cost = waste + r * lost
    return waste, lost, _draws(cost, week), _draws(cost * week.no_stockout, week)


def _best_level(past: Week, order, r: float, life: int) -> float:
    """The multiple of the order that cost least on past weeks, with one delivery.

    Stock that keeps makes overstock cheaper than the newsvendor rule assumes, so its level is
    no longer the best one. At one day the rule stands.
    """
    if life == 1:
        return 1.0
    none = np.zeros_like(order)
    return min(
        MORNING_SHARES, key=lambda s: _cost(past, order, none, DECISION_HOURS[0], s, r, life)
    )


def _row(
    week: Week, one: tuple, order, target, hour: int, share: float, r: float, life: int = 1
) -> dict:
    """One plan's held-out results next to one delivery's, with paired intervals."""
    delivered, extra, waste, lost = _plan(week, order, target, hour, share, life)
    one_waste, one_lost, one_draws, one_clear = one
    cost, one_cost, clear = waste + r * lost, one_waste + r * one_lost, week.no_stockout
    draws, clear_draws = _draws(cost, week), _draws(cost * clear, week)
    demand = week.demand.sum()
    store_days = week.frame.store_id.nunique() * week.demand.shape[1]
    return {
        "ratio": r,
        "decision_hour": hour,
        "arrival_hour": hour + LEAD,
        "morning_share": share,
        "cost": cost.sum() / demand,
        "cost_low": np.percentile(draws, 2.5),
        "cost_high": np.percentile(draws, 97.5),
        "one_delivery_cost": one_cost.sum() / demand,
        "saving": 1 - cost.sum() / one_cost.sum(),
        "saving_low": np.percentile(1 - draws / one_draws, 2.5),
        "saving_high": np.percentile(1 - draws / one_draws, 97.5),
        "saving_no_stockout": 1 - cost[clear].sum() / one_cost[clear].sum(),
        "saving_no_stockout_low": np.percentile(1 - clear_draws / one_clear, 2.5),
        "saving_no_stockout_high": np.percentile(1 - clear_draws / one_clear, 97.5),
        "saved_per_store_day": (one_cost - cost).sum() / store_days,
        "waste": waste.sum() / demand,
        "one_delivery_waste": one_waste.sum() / demand,
        "fill_rate": 1 - lost.sum() / demand,
        "one_delivery_fill_rate": 1 - one_lost.sum() / demand,
        "top_up_share": extra.sum() / delivered.sum(),
    }


def run(
    h: Hours,
    folds: pd.DataFrame,
    held_out: pd.DataFrame,
    ratios=RATIOS,
    headline=HEADLINE,
    life: int = 1,
) -> dict[str, pd.DataFrame]:
    """Score a top-up at every decision hour and cost ratio; only past weeks pick the plan."""
    if headline not in ratios:
        raise ValueError(f"headline ratio {headline:g} is not one of the ratios {ratios}")
    groups = held_out[[*KEYS, "group"]].drop_duplicates()
    past = _week(calibrate(folds, shifts(folds, groups), groups), h)
    week = _week(held_out, h)
    orders = {r: order_at(week.deciles, critical_ratio(r)) for r in ratios}
    levels = {
        r: _best_level(past, order_at(past.deciles, critical_ratio(r)), r, life) for r in ratios
    }
    one = {r: _one_delivery(week, levels[r] * orders[r], r, life) for r in ratios}

    def targets(rest: pd.DataFrame, table: pd.DataFrame, like: Week) -> np.ndarray:
        calibrated = calibrate(rest, table, groups)[CALIBRATED_COLUMNS]
        return calibrated.to_numpy(dtype=float).reshape(like.deciles.shape)

    rows, tables = [], {}
    for hour in DECISION_HOURS:
        past_rest = _rest_of_day(past, hour)
        table = tables[hour] = shifts(past_rest, groups)
        past_targets = targets(past_rest, table, past)
        week_targets = targets(_rest_of_day(week, hour), table, week)
        for r in ratios:
            ratio = critical_ratio(r)
            on_past = order_at(past.deciles, ratio), order_at(past_targets, ratio)
            costs = {s: _cost(past, *on_past, hour, s, r, life) for s in MORNING_SHARES}
            share = min(costs, key=costs.get)
            target = order_at(week_targets, ratio)
            row = _row(week, one[r], orders[r], target, hour, share, r, life)
            rows.append(row | {"past_cost": costs[share], "one_delivery_level": levels[r]})

    plans = pd.DataFrame(rows)
    plans["chosen"] = plans.index.isin(plans.groupby("ratio").past_cost.idxmin())
    sample = pd.DataFrame([{
        "stores": h.ids.store_id.nunique(), "series": len(h.ids),
        "past_days": past.demand.shape[1], "held_out_days": week.demand.shape[1],
        "lead_hours": LEAD, "headline_ratio": headline,
    }])
    best = plans[plans.chosen & (plans.ratio == headline)].iloc[0]
    served = tables[int(best.decision_hour)].assign(
        decision_hour=int(best.decision_hour),
        arrival_hour=int(best.arrival_hour),
        morning_share=best.morning_share,
    )
    return {"plans": plans, "shifts": served, "sample": sample}


def shelf_life(
    h: Hours, folds: pd.DataFrame, forecasts: pd.DataFrame, r: float = HEADLINE, lives=LIVES
) -> pd.DataFrame:
    """The headlines if unsold stock keeps longer, one row per shelf life.

    The fixed policies keep their order levels and count the stock carried over. For the top-up
    and the one delivery it is judged against, past weeks choose the levels again.
    """
    held_out = forecasts[forecasts.model == CLASSICAL]
    week = _week(held_out, h)

    def median(model: str) -> np.ndarray:
        rows = forecasts[forecasts.model == model].sort_values([*KEYS, "dt"])
        return rows.q50.to_numpy(dtype=float).reshape(week.demand.shape)

    policies = {
        "last_week": median(BASELINE),
        "median": median(CLASSICAL),
        "cost_based": order_at(week.deciles, critical_ratio(r)),
    }
    rows = []
    for life in lives:
        plans = run(h, folds, held_out, ratios=(r,), headline=r, life=life)["plans"]
        best = plans[plans.chosen].iloc[0]
        one = {name: _one_delivery(week, q, r, life) for name, q in policies.items()}
        costs = {n: (w + r * lost).sum() / week.demand.sum() for n, (w, lost, *_) in one.items()}
        draws = {name: results[2] for name, results in one.items()}
        versus = 1 - draws["cost_based"] / draws["median"]
        rows.append({
            "life_days": life,
            "last_week_cost": costs["last_week"],
            "median_cost": costs["median"],
            "cost_based_cost": costs["cost_based"],
            "saving_vs_median": 1 - costs["cost_based"] / costs["median"],
            "saving_vs_median_low": np.percentile(versus, 2.5),
            "saving_vs_median_high": np.percentile(versus, 97.5),
            "saving_vs_last_week": 1 - costs["cost_based"] / costs["last_week"],
            "retuned_cost": best.one_delivery_cost,
            "retuned_level": best.one_delivery_level,
            "retuned_waste": best.one_delivery_waste,
            "retuned_fill_rate": best.one_delivery_fill_rate,
            "decision_hour": best.decision_hour,
            "morning_share": best.morning_share,
            "top_up_cost": best.cost,
            "top_up_waste": best.waste,
            "top_up_fill_rate": best.fill_rate,
            "top_up_saving": best.saving,
            "top_up_saving_low": best.saving_low,
            "top_up_saving_high": best.saving_high,
            "top_up_saving_no_stockout": best.saving_no_stockout,
            "saved_per_store_day": best.saved_per_store_day,
        })
    return pd.DataFrame(rows)


def replay(
    h: Hours, orders: pd.DataFrame, actual: pd.DataFrame, r: float = HEADLINE
) -> pd.DataFrame:
    """Score the batch's served plan on the week's demand: one row, as in `run`."""
    calibrated = dict(zip(DECILE_COLUMNS, CALIBRATED_COLUMNS, strict=True))
    frame = orders.merge(actual, on=[*KEYS, "dt"]).rename(columns=calibrated)
    week = _week(
        frame.assign(actual=frame.recovered_demand, in_stock=frame.stockout_hours == 0), h
    )
    level = critical_ratio(r)
    order = order_at(week.deciles, level)
    rest = week.frame[REST_COLUMNS].to_numpy(dtype=float).reshape(week.deciles.shape)
    hour, share = int(orders.decision_hour.iloc[0]), float(orders.morning_share.iloc[0])
    row = _row(week, _one_delivery(week, order, r), order, order_at(rest, level), hour, share, r)
    return pd.DataFrame([row | {"series": len(h.ids), "stores": h.ids.store_id.nunique()}])


def render_report(results: dict[str, pd.DataFrame]) -> str:
    p, s = results["plans"], results["sample"].iloc[0]
    r0 = s.headline_ratio
    best = p[p.chosen & (p.ratio == r0)].iloc[0]
    shown = ["decision_hour", "arrival_hour", "morning_share", "past_cost", "cost", "saving"]
    life = results["shelf_life"]

    def by_life(column: str) -> str:
        values = zip(life.life_days, life[column], strict=True)
        return ", ".join(f"{v:.0%} at {d} day{'s' if d > 1 else ''}" for d, v in values)

    parts = [
        "# A second delivery",
        "One delivery commits the whole day's order before opening. Here the morning carries a "
        "share of that order; at the decision hour the store counts what is left and orders up "
        f"to a target for the rest of the day, delivered {int(s.lead_hours)} hours later (an "
        "assumption: the data has no delivery times). The target is the day's calibrated "
        "forecast, scaled to each line's usual share of demand still to come and calibrated "
        "again on past weeks. Waste and lost sales are charged as in `reports/orders.md`, using "
        "LightGBM's forecasts. The decision hour and morning share are chosen on "
        f"{int(s.past_days)} past days; the held-out week only scores them. Intervals come from "
        "1,000 paired bootstrap resamples of series. Generated by "
        "`uv run python -m fresh_replenish.decisions.topup`.",
        "## Headline",
        f"- **Best plan at r = {r0:g}:** order at {int(best.decision_hour)}:00 for a "
        f"{int(best.arrival_hour)}:00 delivery, with {best.morning_share:.0%} of the one-delivery "
        f"order sent in the morning. It costs {best.cost:.3f} (95% CI {best.cost_low:.3f} to "
        f"{best.cost_high:.3f}) per unit of demand, {best.saving:.0%} less than one delivery "
        f"({best.one_delivery_cost:.3f}; 95% CI of the saving {best.saving_low:.0%} to "
        f"{best.saving_high:.0%}).",
        f"- **Where it comes from:** waste falls from {best.one_delivery_waste:.3f} to "
        f"{best.waste:.3f} per unit of demand, and the fill rate moves from "
        f"{best.one_delivery_fill_rate:.1%} to {best.fill_rate:.1%}. "
        f"{best.top_up_share:.0%} of delivered units come on the top-up.",
        "- **What it leaves out:** the saving does not count what the second delivery itself "
        f"costs. At r = {r0:g} it is worth {best.saved_per_store_day:.1f} wasted units per store "
        "per day, so a delivery pays only below that.",
        f"- **Days with no stockout:** the saving is {best.saving_no_stockout:.0%} (95% CI "
        f"{best.saving_no_stockout_low:.0%} to {best.saving_no_stockout_high:.0%}). Their hourly "
        "demand is fully observed, while on other days the afternoon is recovered, not seen. "
        "They are also a selected set of days, so the gap is not all down to recovery.",
        "## Best plan at each cost ratio",
        md_table(p[p.chosen].drop(columns="chosen")),
        f"## Every decision hour at r = {r0:g}",
        md_table(p.loc[p.ratio == r0, shown]),
        "## If stock keeps longer",
        "Everything above throws unsold stock away the same night. The data has no shelf lives, "
        f"so this lets unsold units keep for up to {int(life.life_days.max())} days at "
        f"r = {r0:g}, oldest sold first. Each policy counts the stock it carries. The week runs "
        "twice and the second pass is scored, so the shelf starts with normal stock.",
        f"- **Cost-based orders against the most accurate forecast:** the saving is "
        f"{by_life('saving_vs_median')}.",
        f"- **Cost-based orders against last week's demand:** "
        f"{by_life('saving_vs_last_week')}.",
        "- **What a second delivery is worth:** "
        + ", ".join(
            f"{v:.1f} wasted units per store per day at {d} day{'s' if d > 1 else ''}"
            for d, v in zip(life.life_days, life.saved_per_store_day, strict=True)
        )
        + ".",
        "- **A second delivery against one:** "
        f"{by_life('top_up_saving')}. Stock that keeps makes overstock cheaper, so the newsvendor "
        "level is no longer the best one: for this comparison, past weeks choose the level of "
        "the one delivery again, as well as the top-up's hour and share.",
        md_table(life),
        "## Sample",
        md_table(results["sample"]),
    ]
    return "\n\n".join(parts) + "\n"


def main(
    raw_dir: Path = RAW_DIR, out_dir: Path = REPORTS_DIR, processed_dir: Path = PROCESSED_DIR
) -> int:
    steps = {"forecast_folds": "forecasting.evaluate", "forecast_eval": "uncertainty.conformal"}
    for name, module in steps.items():
        if not (processed_dir / f"{name}.parquet").exists():
            print(
                f"{name}.parquet not found. Run: uv run python -m fresh_replenish.{module}",
                file=sys.stderr,
            )
            return 2
    paths = local_paths(raw_dir)
    if paths is None:
        return 2
    con = connect(*paths)
    stores = sample_stores(con, evaluate.SAMPLE_SHARE, np.random.default_rng(SEED))
    hours = load_hours(con, stores=stores, splits=("train", "eval"))
    folds = pd.read_parquet(
        processed_dir / "forecast_folds.parquet", filters=[("model", "==", CLASSICAL)]
    )
    forecasts = pd.read_parquet(
        processed_dir / "forecast_eval.parquet", filters=[("model", "in", [CLASSICAL, BASELINE])]
    )
    results = run(hours, folds, forecasts[forecasts.model == CLASSICAL])
    results["shelf_life"] = shelf_life(hours, folds, forecasts)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / "tables" / f"topup_{name}.csv", index=False)
    report = out_dir / "topup.md"
    report.write_text(render_report(results), encoding="utf-8")
    print(f"Report written to {report}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
