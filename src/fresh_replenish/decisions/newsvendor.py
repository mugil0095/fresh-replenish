"""Cost-based orders (spec step 6): newsvendor stock levels and a policy simulation.

The data has no prices, so a wasted unit costs 1 and a lost sale costs r, swept over RATIOS.
Each order covers one day: unsold units are wasted (one-day shelf life) and unmet demand is lost.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fresh_replenish.config import PROCESSED_DIR, REPORTS_DIR
from fresh_replenish.forecasting.data import BASELINE
from fresh_replenish.metrics import CALIBRATED_COLUMNS, DECILE_COLUMNS, DECILES, bootstrap
from fresh_replenish.report import md_table

RATIOS = (0.5, 1.0, 2.0, 4.0, 9.0)  # lost-sale cost / waste cost
HEADLINE = 9.0  # availability-first, as quick commerce runs today (docs/decisions.md)


def critical_ratio(r: float) -> float:
    """The newsvendor service level: the chance of not running out that minimises cost."""
    return r / (1 + r)


def order_at(q: np.ndarray, ratio: float) -> np.ndarray:
    """Order at a critical ratio, interpolated between the deciles on q's last axis."""
    if not DECILES[0] - 1e-9 <= ratio <= DECILES[-1] + 1e-9:
        raise ValueError(f"critical ratio {ratio:.2f} is outside the deciles 0.1-0.9")
    k = min(int(np.searchsorted(DECILES, ratio + 1e-9)) - 1, len(DECILES) - 2)
    w = (ratio - DECILES[k]) / (DECILES[k + 1] - DECILES[k])
    return q[..., k] * (1 - w) + q[..., k + 1] * w


def simulate(order, demand, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-series sums of cost (waste + r x lost sales), order minus demand, and demand."""
    waste, lost = np.maximum(order - demand, 0), np.maximum(demand - order, 0)
    return (waste + r * lost).sum(axis=1), (order - demand).sum(axis=1), demand.sum(axis=1)


def run(forecasts: pd.DataFrame, ratios=RATIOS, headline=HEADLINE) -> dict[str, pd.DataFrame]:
    """Simulate every ordering policy at every cost ratio on the held-out week."""
    if headline not in ratios:
        raise ValueError(f"headline ratio {headline:g} is not one of the ratios {ratios}")
    keys = ["store_id", "product_id", "dt"]
    frame = forecasts.sort_values(["model", *keys])
    base = frame[frame.model == BASELINE]
    ids = base.drop_duplicates(["store_id", "product_id"])
    n = len(ids)
    demand = base.actual.to_numpy().reshape(n, -1)

    def block(model, cols):
        part = frame[frame.model == model]
        if not np.array_equal(part[keys].to_numpy(), base[keys].to_numpy()):
            raise ValueError(f"{model} must forecast every series and day that {BASELINE} does")
        return part[cols].to_numpy().reshape(n, -1, len(cols))

    calibrated = forecasts.dropna(subset=CALIBRATED_COLUMNS).model.unique()
    deciles = {m: block(m, CALIBRATED_COLUMNS) for m in calibrated}
    medians = {m: block(m, DECILE_COLUMNS)[..., len(DECILES) // 2] for m in calibrated}
    lowest = min(medians, key=lambda m: np.abs(medians[m] - demand).sum())  # lowest WAPE
    fixed = {BASELINE: block(BASELINE, ["q50"])[..., 0], f"{lowest} median": medians[lowest]}
    rows, draws, at_headline = [], {}, {}
    for r in ratios:
        ratio = critical_ratio(r)
        for name, order in (fixed | {m: order_at(q, ratio) for m, q in deciles.items()}).items():
            sums = simulate(order, demand, r)
            point, draws[r, name] = bootstrap(sums)
            rows.append({
                "ratio": r,
                "critical_ratio": ratio,
                "policy": name,
                "cost": point["wape"],  # cost per unit of demand
                "cost_low": point["wape_low"],
                "cost_high": point["wape_high"],
                "order_bias": point["wpe"],
                "waste_share": np.maximum(order - demand, 0).sum() / order.sum(),
                "fill_rate": np.minimum(order, demand).sum() / demand.sum(),
                "stockout_days": float((order < demand).mean()),
            })
            if r == headline:
                at_headline[name] = sums[0]

    policies = pd.DataFrame(rows)
    keys = list(zip(policies.ratio, policies.policy, strict=True))
    cost = dict(zip(keys, policies.cost, strict=True))
    cheapest = policies.loc[policies.groupby("ratio").cost.idxmin()].set_index("ratio").policy
    policies["saving_vs_naive"] = [1 - cost[k] / cost[k[0], BASELINE] for k in keys]
    for col, ref in (("vs_lowest_error", lambda r: lowest), ("vs_cheapest", cheapest.get)):
        diffs = [draws[k] - draws[k[0], ref(k[0])] for k in keys]  # same resamples: paired
        policies[col] = [cost[k] - cost[k[0], ref(k[0])] for k in keys]
        policies[col + "_low"] = [np.percentile(d, 2.5) for d in diffs]
        policies[col + "_high"] = [np.percentile(d, 97.5) for d in diffs]

    per_store = pd.DataFrame(at_headline).groupby(ids.store_id.to_numpy()).sum()
    wins = per_store.idxmin(axis=1).value_counts().reindex(per_store.columns, fill_value=0)
    stores = pd.DataFrame({
        "policy": wins.index, "stores": wins.to_numpy(), "share": wins.to_numpy() / len(per_store)
    })
    sample = pd.DataFrame([{
        "stores": len(per_store), "series": n, "days": demand.shape[1],
        "ratios": ", ".join(f"{r:g}" for r in ratios), "headline_ratio": headline,
        "lowest_error_model": lowest,
    }])
    return {"policies": policies, "stores": stores, "sample": sample}


def render_report(results: dict[str, pd.DataFrame]) -> str:
    p = results["policies"]
    s = results["sample"].iloc[0]
    r0 = s.headline_ratio
    head = p[p.ratio == r0].set_index("policy")
    best = head.cost.idxmin()
    lowest = s.lowest_error_model
    wins = results["stores"].set_index("policy").stores

    def cost(row):
        return f"{row.cost:.3f} (95% CI {row.cost_low:.3f} to {row.cost_high:.3f})"

    misses = []
    for r, part in p.groupby("ratio"):
        part = part.set_index("policy")
        cheapest, ref = part.cost.idxmin(), part.loc[lowest]
        if cheapest != lowest and ref.vs_cheapest_low > 0:
            misses.append(
                f"r = {r:g}: {cheapest} is cheaper by {ref.vs_cheapest:.3f} "
                f"(95% CI {ref.vs_cheapest_low:.3f} to {ref.vs_cheapest_high:.3f})"
            )
    store_wins = ", ".join(f"{name} in {k}" for name, k in wins.items() if k)

    parts = [
        "# Cost-based orders",
        "Each policy orders for every series and day of the held-out week. The data has no "
        "prices, so a wasted unit costs 1 and a lost sale costs r; unsold stock is thrown away "
        "at the end of the day and missed sales are not made up later. Demand is recovered as "
        "in Phase 3. Cost is per unit of demand. Intervals come from 1,000 bootstrap resamples "
        "of series, the same resamples for every policy, so differences are paired. Seasonal "
        "naive orders last week's same-day sales; the median policy orders the median of the "
        "model with the lowest held-out WAPE; every other policy orders that model's "
        "calibrated forecast at the critical ratio r / (1 + r), the newsvendor rule. "
        f"r = {r0:g} is the headline because quick commerce now runs availability-first "
        "(see `docs/decisions.md`). Generated by "
        "`uv run python -m fresh_replenish.decisions.newsvendor`.",
        "## Headline",
        f"- **Cheapest at r = {r0:g}:** {best}, at {cost(head.loc[best])} per unit of demand, "
        f"{head.loc[best, 'saving_vs_naive']:.0%} less than ordering last week's sales.",
        f"- **Lowest-error model ({lowest}):** its cost-based orders cost "
        f"{cost(head.loc[lowest])}; ordering its median instead costs "
        f"{head.loc[f'{lowest} median', 'cost']:.3f}.",
        "- **Where the lowest-error model is not the cheapest decision** (paired interval above "
        "0): " + ("; ".join(misses) or f"nowhere: {lowest}'s orders are also the cheapest") + ".",
        f"- **Across stores at r = {r0:g}:** the cheapest policy is {store_wins} "
        f"of {int(s.stores)} stores.",
        "## Policies",
        md_table(p),
        f"## Cheapest policy per store at r = {r0:g}",
        md_table(results["stores"]),
        "## Sample",
        md_table(results["sample"]),
    ]
    return "\n\n".join(parts) + "\n"


def main(out_dir: Path = REPORTS_DIR, processed_dir: Path = PROCESSED_DIR) -> int:
    path = processed_dir / "forecast_eval.parquet"
    if not path.exists():
        print(
            f"{path.name} not found. Run: uv run python -m fresh_replenish.uncertainty.conformal",
            file=sys.stderr,
        )
        return 2
    results = run(pd.read_parquet(path))
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / "tables" / f"orders_{name}.csv", index=False)
    report = out_dir / "orders.md"
    report.write_text(render_report(results), encoding="utf-8")
    print(f"Report written to {report}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
