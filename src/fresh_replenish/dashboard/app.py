"""Fresh-food replenishment dashboard.

Run it with `uv run streamlit run src/fresh_replenish/dashboard/app.py` after the pipeline has
written its outputs (see the README).
"""

import altair as alt
import pandas as pd
import streamlit as st

from fresh_replenish.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from fresh_replenish.dashboard import data
from fresh_replenish.decisions.newsvendor import HEADLINE, RATIOS, critical_ratio, order_at
from fresh_replenish.forecasting.data import BASELINE
from fresh_replenish.metrics import DECILE_COLUMNS

TABLES = REPORTS_DIR / "tables"
RECORDS = PROCESSED_DIR.parent / "decisions" / "eval_records.jsonl"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
BAND = "#cde2fb"  # blue step 100: the forecast range
GRAY = "#52514e"  # secondary ink: the demand history

st.set_page_config(page_title="Fresh replenishment", layout="wide")


@st.cache_data
def load() -> dict:
    return {
        "policies": pd.read_csv(TABLES / "orders_policies.csv"),
        "forecasts": pd.read_parquet(PROCESSED_DIR / "forecast_eval.parquet"),
        "categories": data.categories(RAW_DIR / "data" / "train.parquet"),
        "stockouts": pd.read_csv(TABLES / "stockout_by_hour.csv"),
        "coverage": pd.read_csv(TABLES / "calibration_coverage.csv").set_index("model"),
        "agent": pd.read_csv(TABLES / "agent_summary_gpt-oss-120b.csv").set_index("type"),
    }


@st.cache_data
def batch_orders() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "orders.parquet")


def stacked(frame: pd.DataFrame, by: str, title: str) -> alt.Chart:
    """Waste and lost-sales cost per group as stacked horizontal bars, costliest on top."""
    long = frame.melt(
        id_vars=[by, "total"], value_vars=["waste_cost", "lost_sales_cost"],
        var_name="kind", value_name="cost",
    ).replace({"kind": {"waste_cost": "Waste", "lost_sales_cost": "Lost sales"}})
    order = frame[by].astype(str).tolist()
    return alt.Chart(long.assign(**{by: long[by].astype(str)}), title=title).mark_bar(
        cornerRadiusEnd=4, stroke="#ffffff", strokeWidth=2
    ).encode(
        y=alt.Y(f"{by}:N", sort=order, title=None),
        x=alt.X("sum(cost):Q", title="Cost (wasted units = 1)"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Waste", "Lost sales"],
                                                  range=[BLUE, ORANGE]), title=None),
        tooltip=[alt.Tooltip(f"{by}:N"), alt.Tooltip("kind:N"),
                 alt.Tooltip("cost:Q", format=",.1f")],
    )


try:
    d = load()
except FileNotFoundError as missing:
    st.error(f"Pipeline output not found ({missing.filename}). Run the steps in the README first.")
    st.stop()

h = data.headline(d["policies"])
st.title("Fresh-food replenishment for quick-commerce dark stores")
st.markdown(
    f"### Cost-based orders cut simulated waste + lost-sales cost by **{h['vs_median']:.0%}** "
    "compared with ordering the most accurate forecast."
)
st.caption(
    "Held-out week (Jun 26 to Jul 2, 2024), 4,928 store-product series in 90 dark stores. "
    f"Assumes a lost sale costs {h['ratio']:g}x a wasted unit, because the data has no prices; "
    "the Ordering policies tab shows other ratios."
)
cols = st.columns(4)
cols[0].metric("vs most accurate forecast", f"-{h['vs_median']:.0%}")
cols[1].metric("vs last week's sales", f"-{h['vs_naive']:.0%}")
cols[2].metric(
    "Demand in forecast range", f"{d['coverage'].loc[h['model'], 'coverage_after']:.0%}",
    help="Share of actual demand inside the calibrated 10-90% range; the target is 80%.",
)
cols[3].metric(
    "Agent numbers from tools", f"{d['agent'].loc['all', 'numbers_match']:.0%}",
    help=f"Correct tool choice: {d['agent'].loc['all', 'tool_choice']:.0%} of 150 runs.",
)

money, policy_tab, forecast_tab, order_tab, agent_tab = st.tabs(
    ["Where money is lost", "Ordering policies", "Forecasts", "Orders", "Agent decisions"]
)

with money:
    ratio = st.select_slider(
        "A lost sale costs this many wasted units", options=list(RATIOS), value=HEADLINE
    )
    st.caption(f"Cost-based orders from {h['model']}; held-out week.")
    left, right = st.columns(2)
    by_category = data.losses(d["forecasts"], d["categories"], ratio)
    by_store = data.losses(d["forecasts"], d["categories"], ratio, by="store_id")
    left.altair_chart(stacked(by_category.head(12), "first_category_id",
                              "Costliest product categories"), use_container_width=True)
    right.altair_chart(stacked(by_store.head(12), "store_id", "Costliest stores"),
                       use_container_width=True)
    hours = alt.Chart(d["stockouts"], title="Share of hours out of stock, by hour of day").mark_bar(
        color=BLUE, cornerRadiusEnd=4
    ).encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("stockout_share:Q", title="Out of stock", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("hour:O"), alt.Tooltip("stockout_share:Q", format=".1%")],
    )
    st.altair_chart(hours, use_container_width=True)
    st.caption("Evening demand is the most hidden: stockouts climb through the day.")

with policy_tab:
    shown = [BASELINE, f"{h['model']} median", h["model"]]
    labels = {BASELINE: "Reorder last week's sales", f"{h['model']} median":
              "Order the most accurate forecast", h["model"]: "Cost-based order"}
    curve = d["policies"][d["policies"].policy.isin(shown)].assign(
        policy=lambda f: f.policy.map(labels)
    )
    line = alt.Chart(curve, title="Cost per unit of demand, by cost ratio").mark_line(
        point=alt.OverlayMarkDef(size=64), strokeWidth=2
    ).encode(
        x=alt.X("ratio:Q", scale=alt.Scale(type="log", domain=[0.5, 9], nice=False),
                axis=alt.Axis(values=list(RATIOS)),
                title="A lost sale costs this many wasted units"),
        y=alt.Y("cost:Q", title="Cost per unit of demand"),
        color=alt.Color("policy:N", scale=alt.Scale(domain=list(labels.values()),
                                                    range=[ORANGE, AQUA, BLUE]), title=None,
                        legend=alt.Legend(labelLimit=320, orient="bottom")),
        tooltip=[alt.Tooltip("policy:N"), alt.Tooltip("ratio:Q"),
                 alt.Tooltip("cost:Q", format=".3f"),
                 alt.Tooltip("fill_rate:Q", format=".1%", title="Fill rate")],
    )
    st.altair_chart(line, use_container_width=True)
    st.caption("When lost sales cost more than waste, ordering the most accurate number is "
               "the wrong decision: the cost-based order is cheaper at every ratio above 1.")
    st.dataframe(
        d["policies"][d["policies"].ratio == HEADLINE][
            ["policy", "cost", "cost_low", "cost_high", "fill_rate", "waste_share"]
        ].round(3),
        hide_index=True,
    )

with forecast_tab:
    forecasts = d["forecasts"]
    left, right = st.columns(2)
    store = left.selectbox("Store", sorted(forecasts.store_id.unique()))
    product = right.selectbox(
        "Product", sorted(forecasts.loc[forecasts.store_id == store, "product_id"].unique())
    )
    week = forecasts[(forecasts.model == h["model"]) & (forecasts.store_id == store)
                     & (forecasts.product_id == product)]
    past = data.history(PROCESSED_DIR / "recovered_demand.parquet", int(store), int(product))
    band = alt.Chart(week).mark_area(color=BAND).encode(
        x=alt.X("dt:T", title=None), y=alt.Y("cal_q10:Q", title="Daily demand"), y2="cal_q90:Q",
        tooltip=[alt.Tooltip("dt:T"), alt.Tooltip("cal_q10:Q", format=".2f", title="10th pct"),
                 alt.Tooltip("cal_q90:Q", format=".2f", title="90th pct")],
    )
    median = alt.Chart(week).mark_line(color=BLUE, strokeWidth=2).encode(x="dt:T", y="cal_q50:Q")
    actual = alt.Chart(week).mark_point(color=ORANGE, filled=True, size=64).encode(
        x="dt:T", y="actual:Q",
        tooltip=[alt.Tooltip("dt:T"), alt.Tooltip("actual:Q", format=".2f", title="Actual")],
    )
    before = alt.Chart(past).mark_line(color=GRAY, strokeWidth=2).encode(
        x="dt:T", y="recovered_demand:Q",
        tooltip=[alt.Tooltip("dt:T"), alt.Tooltip("recovered_demand:Q", format=".2f")],
    )
    st.altair_chart((band + before + median + actual).properties(
        title="Recovered demand (gray), forecast range (blue band), median (blue), actual (orange)"
    ), use_container_width=True)

with order_tab:
    orders = batch_orders()
    left, right = st.columns(2)
    store = left.selectbox("Store ", sorted(orders.store_id.unique()))
    ratio = right.select_slider("Cost ratio ", options=list(RATIOS), value=HEADLINE)
    lines = orders[orders.store_id == store].copy()
    lines["order"] = order_at(lines[DECILE_COLUMNS].to_numpy(), critical_ratio(ratio))
    st.caption(f"Nightly batch as of {lines.as_of.iloc[0]}: calibrated LightGBM forecasts "
               f"ordered at the {critical_ratio(ratio):.0%} level.")
    st.dataframe(
        lines[["product_id", "dt", "order", "q10", "q50", "q90"]].rename(
            columns={"dt": "date", "q10": "low", "q50": "median", "q90": "high"}
        ).round(2), hide_index=True,
    )

with agent_tab:
    agent = d["agent"]
    st.markdown(
        f"On 150 scripted runs (Ollama Cloud gpt-oss:120b) the agent chose the right tools "
        f"**{agent.loc['all', 'tool_choice']:.0%}** of the time, every stated number came from "
        f"a tool (**{agent.loc['all', 'numbers_match']:.0%}**), and every over-limit order "
        f"went to a planner (**{agent.loc['limit', 'handoff_correct']:.0%}**)."
    )
    st.dataframe(data.records(RECORDS), hide_index=True)
    st.caption("Decision records written by the agent during the evaluation.")
