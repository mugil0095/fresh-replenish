"""A second delivery: count the shelf at midday and top up for the rest of the day.

One delivery (`newsvendor.py`) commits the whole day's order before opening. Here the morning
carries a share of that order; at the decision hour the store counts what is left and orders up
to a calibrated target for the rest of the day, delivered LEAD hours later. Days end as before:
unsold units are wasted and missed sales are not made up.
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
    simulate,
)
from fresh_replenish.forecasting import evaluate
from fresh_replenish.forecasting.data import CLASSICAL
from fresh_replenish.imputation.data import Hours, load_hours
from fresh_replenish.imputation.statistical import profile_recover
from fresh_replenish.metrics import CALIBRATED_COLUMNS, DECILE_COLUMNS, SEED, bootstrap
from fresh_replenish.report import md_table
from fresh_replenish.uncertainty.conformal import KEYS, calibrate, shifts

OPENING = 6  # the hourly arrays start at 6:00
LEAD = 2  # hours from order to delivery: an assumption, the data has no delivery times
DECISION_HOURS = range(8, 19)
MORNING_SHARES = np.round(np.arange(4, 11) / 10, 1)  # of the one-delivery order


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


def top_up(target, morning, seen):
    """Units to order up to the target, counting what is left on the shelf at the decision hour."""
    return np.maximum(target - np.maximum(morning - seen, 0), 0)


def serve(morning, extra, early, late):
    """Waste and lost sales of a day: `early` demand comes before the top-up lands, `late` after."""
    stock = np.maximum(morning - early, 0) + extra
    return np.maximum(stock - late, 0), np.maximum(early - morning, 0) + np.maximum(late - stock, 0)


@dataclass
class Week:
    frame: pd.DataFrame  # one row per series and day, sorted by series, then day
    deciles: np.ndarray  # (series, days, 9): the calibrated daily forecast
    demand: np.ndarray  # (series, days)
    hourly: np.ndarray  # (series, days, hours): the demand, hour by hour
    left: np.ndarray  # (series, hours): usual share of demand left after each hour


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
        hourly=spread * demand[..., None],  # the same daily totals as the one-delivery simulation
        left=still_to_come(hourly_demand(h, int(days.argmax()))),
    )


def _rest_of_day(week: Week, hour: int) -> pd.DataFrame:
    """A forecast frame for the demand left at the decision hour: the day's deciles, scaled."""
    t = hour - OPENING
    frame = week.frame[["model", *KEYS]].assign(actual=week.hourly[..., t:].sum(axis=-1).ravel())
    scaled = week.deciles * week.left[:, None, t - 1, None]
    frame[DECILE_COLUMNS] = scaled.reshape(-1, len(DECILE_COLUMNS))
    return frame


def _plan(week: Week, order, target, hour: int, share: float) -> tuple:
    """Units delivered, topped up, wasted and lost on every line-day under one plan."""
    t = hour - OPENING
    morning = share * order
    extra = top_up(target, morning, week.hourly[..., :t].sum(axis=-1))
    early = week.hourly[..., : t + LEAD].sum(axis=-1)
    return morning + extra, extra, *serve(morning, extra, early, week.demand - early)


def _cost(week: Week, order, target, hour: int, share: float, r: float) -> float:
    *_, waste, lost = _plan(week, order, target, hour, share)
    return (waste + r * lost).sum() / week.demand.sum()


def _row(week: Week, order, target, hour: int, share: float, r: float, store_days: int) -> dict:
    """One plan's held-out results next to one delivery's, with paired intervals."""
    delivered, extra, waste, lost = _plan(week, order, target, hour, share)
    demand = week.demand
    no_stockout = week.frame.in_stock.to_numpy().reshape(demand.shape)
    cost_two = waste + r * lost
    cost_one = np.maximum(order - demand, 0) + r * np.maximum(demand - order, 0)
    one, one_draws = bootstrap(simulate(order, demand, r))
    two, two_draws = bootstrap(
        (cost_two.sum(axis=1), (delivered - demand).sum(axis=1), demand.sum(axis=1))
    )
    saving_draws = 1 - two_draws / one_draws  # the same resamples, so the saving is paired
    return {
        "ratio": r,
        "decision_hour": hour,
        "arrival_hour": hour + LEAD,
        "morning_share": share,
        "cost": two["wape"],
        "cost_low": two["wape_low"],
        "cost_high": two["wape_high"],
        "one_delivery_cost": one["wape"],
        "saving": 1 - two["wape"] / one["wape"],
        "saving_low": np.percentile(saving_draws, 2.5),
        "saving_high": np.percentile(saving_draws, 97.5),
        "saving_no_stockout": 1 - cost_two[no_stockout].sum() / cost_one[no_stockout].sum(),
        "saved_per_store_day": (cost_one.sum() - cost_two.sum()) / store_days,
        "waste": waste.sum() / demand.sum(),
        "one_delivery_waste": np.maximum(order - demand, 0).sum() / demand.sum(),
        "fill_rate": 1 - lost.sum() / demand.sum(),
        "one_delivery_fill_rate": np.minimum(order, demand).sum() / demand.sum(),
        "topped_up": float((extra > 0).mean()),
    }


def run(
    h: Hours, folds: pd.DataFrame, held_out: pd.DataFrame, ratios=RATIOS, headline=HEADLINE
) -> dict[str, pd.DataFrame]:
    """Score a top-up at every decision hour and cost ratio; only past weeks pick the plan."""
    if headline not in ratios:
        raise ValueError(f"headline ratio {headline:g} is not one of the ratios {ratios}")
    groups = held_out[[*KEYS, "group"]].drop_duplicates()
    past = _week(calibrate(folds, shifts(folds, groups), groups), h)
    week = _week(held_out, h)
    store_days = h.ids.store_id.nunique() * week.demand.shape[1]

    def targets(rest: pd.DataFrame, table: pd.DataFrame, like: Week) -> np.ndarray:
        calibrated = calibrate(rest, table, groups)[CALIBRATED_COLUMNS]
        return calibrated.to_numpy(dtype=float).reshape(like.deciles.shape)

    rows = []
    for hour in DECISION_HOURS:
        past_rest = _rest_of_day(past, hour)
        table = shifts(past_rest, groups)
        past_targets = targets(past_rest, table, past)
        week_targets = targets(_rest_of_day(week, hour), table, week)
        for r in ratios:
            ratio = critical_ratio(r)
            on_past = order_at(past.deciles, ratio), order_at(past_targets, ratio)
            costs = {s: _cost(past, *on_past, hour, s, r) for s in MORNING_SHARES}
            share = min(costs, key=costs.get)
            order, target = order_at(week.deciles, ratio), order_at(week_targets, ratio)
            row = _row(week, order, target, hour, share, r, store_days)
            rows.append(row | {"past_cost": costs[share]})

    plans = pd.DataFrame(rows)
    plans["chosen"] = plans.index.isin(plans.groupby("ratio").past_cost.idxmin())
    sample = pd.DataFrame([{
        "stores": h.ids.store_id.nunique(), "series": len(h.ids),
        "past_days": past.demand.shape[1], "held_out_days": week.demand.shape[1],
        "lead_hours": LEAD, "headline_ratio": headline,
    }])
    return {"plans": plans, "sample": sample}


def render_report(results: dict[str, pd.DataFrame]) -> str:
    p, s = results["plans"], results["sample"].iloc[0]
    r0 = s.headline_ratio
    best = p[p.chosen & (p.ratio == r0)].iloc[0]
    shown = ["decision_hour", "arrival_hour", "morning_share", "past_cost", "cost", "saving"]
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
        f"{best.topped_up:.0%} of line-days get a top-up.",
        "- **What it leaves out:** the saving does not count what the second delivery itself "
        f"costs. It is worth {best.saved_per_store_day:.1f} wasted units per store per day, so "
        "a delivery pays only below that.",
        f"- **Days with no stockout:** the saving is {best.saving_no_stockout:.0%}. Their hourly "
        "demand is fully observed; on other days the afternoon is recovered, not seen.",
        "## Best plan at each cost ratio",
        md_table(p[p.chosen].drop(columns="chosen")),
        f"## Every decision hour at r = {r0:g}",
        md_table(p.loc[p.ratio == r0, shown]),
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
    folds, held_out = (
        pd.read_parquet(processed_dir / f"{name}.parquet", filters=[("model", "==", CLASSICAL)])
        for name in steps
    )
    results = run(hours, folds, held_out)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / "tables" / f"topup_{name}.csv", index=False)
    report = out_dir / "topup.md"
    report.write_text(render_report(results), encoding="utf-8")
    print(f"Report written to {report}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
