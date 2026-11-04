# Demo script (3–5 minutes)

**Before recording**
- Run the pipeline (see the README), so `data/processed/` and `reports/` are filled.
- Put `OLLAMA_API_KEY` in `.env`.
- Open two terminals.

## 0:00 The problem (30 s)
Open `docs/summary.md`. Read the headline: ordering by cost cut waste + lost-sales cost by 56%. Then make the point that stockouts hide demand, so forecasts keep ordering too little.

## 0:30 The dashboard (2 min)
`uv run streamlit run src/fresh_replenish/dashboard/app.py`
1. **Headline cards:** −56%, −55%, 80% of demand inside the range, 100% of the agent's answers passing the number check.
2. **Where money is lost:** move the slider from 9 to 1, and show the cost moving from waste to lost sales. Point at the stockouts-by-hour chart (evenings).
3. **Ordering policies:** the cost-based order is the cheapest line at every ratio. The "most accurate forecast" line climbs fastest and ends above "reorder last week's demand" at 9×.
4. **Forecasts:** pick a store and product. Show the history, the forecast range and the actual demand.

## 2:30 The agent (1.5 min)
1. A what-if:
   `uv run python -m fresh_replenish.agent.chat "What would demand be for product 118 at store 434 on 2024-07-01 if it is on promotion?"`
   It says the answer is a model-based estimate.
2. A risky order:
   `uv run python -m fresh_replenish.agent.chat "Place an order of 6 units of product 4 for store 57 on 2024-07-02."`
   The agent pauses for you, the planner. Answer `n`, and it doesn't record the order.

## 4:00 Production (30 s)
`docker compose up`. Show the API with `curl "localhost:8000/reorder?store_id=0"`, the dashboard on :8501 and MLflow runs on :5000. Close on the test count and CI.
