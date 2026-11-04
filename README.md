# fresh-replenish

**Ordering fresh food by cost instead of by the most accurate forecast cut simulated waste + lost-sales cost by 56%** on a held-out week: 4,928 store-product lines (538 products) in 90 quick-commerce dark stores, with a lost sale assumed to cost 9× a wasted unit. The [one-page summary](docs/summary.md) gives the business view.

| What | Result |
|---|---|
| Demand hidden by stockouts, recovered | +28% on all days, +62% on stockout days |
| Forecast error (WAPE), held-out week | LightGBM 32.5%; Chronos-2 33.3% zero-shot; reorder last week 41.7% |
| Forecast ranges that hold | 80% of demand inside the calibrated 10–90% range (target 80%) |
| Cost of orders, against the most accurate forecast's median | −56% (−55% against reordering last week's demand) |
| Nightly batch / API | 50,000 series in 11.5 min; p95 7.3 ms per store request |
| AI agent (50 questions × 3 runs) | 100% right tools; every final answer passed the number checker; every over-limit order went to a human |

The project:

1. recovers the demand that stockouts hide;
2. forecasts it with time-series foundation models and LightGBM;
3. turns forecasts into calibrated, cost-based daily orders;
4. explains each order with an AI agent that hands risky ones to a human.

**Status:** all 8 phases are built.
- **Data foundation:** the data is verified; see the [data audit](reports/data_audit.md).
- **Demand drivers:** see [what drives demand](reports/demand_drivers.md).
- **Hidden demand:** see [recovering hidden demand](reports/demand_recovery.md).
- **Forecasting:** see [forecasting daily demand](reports/forecasting.md).
- **Uncertainty and orders:** see [calibrated forecast ranges](reports/calibration.md) and [cost-based orders](reports/orders.md).
- **Service:** see [batch and API](reports/service.md) and [monitoring](reports/monitoring.md).
- **Agent:** see the [agent evaluation](reports/agent_gpt-oss-120b.md).
- **Communication:** a Streamlit dashboard (`uv run streamlit run src/fresh_replenish/dashboard/app.py`) and a [demo script](docs/demo.md).

Each choice is explained in [decisions](docs/decisions.md). What didn't work is in [negative results](docs/negative_results.md), what the results can't claim is in [limitations](docs/limitations.md), and the dataset is described in the [data card](docs/data_card.md).

## Run it

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m fresh_replenish.data.download   # 115 MB, pinned dataset revision
uv run python -m fresh_replenish.data.audit      # writes reports/data_audit.md
uv run python -m fresh_replenish.analysis.drivers  # writes reports/demand_drivers.md (~1.5 min)
uv run python -m fresh_replenish.imputation.recovery  # writes reports/demand_recovery.md (~45 min on a GTX 1650)
uv run python -m fresh_replenish.forecasting.evaluate  # writes reports/forecasting.md (~10 min; ~2.5 GB of model weights the first time)
uv run python -m fresh_replenish.uncertainty.conformal  # writes reports/calibration.md (~4 min)
uv run python -m fresh_replenish.decisions.newsvendor   # writes reports/orders.md (seconds)
uv run python -m fresh_replenish.serving.batch          # nightly orders for all 50,000 series (~12 min)
uv run uvicorn --factory fresh_replenish.serving.api:create_app  # GET /reorder?store_id=0
uv run python -m fresh_replenish.serving.latency        # writes reports/service.md
uv run python -m fresh_replenish.monitoring.drift       # writes reports/monitoring.md
uv run streamlit run src/fresh_replenish/dashboard/app.py  # the dashboard on :8501
docker compose up                                       # the API on :8000, dashboard on :8501, MLflow on :5000
FRESH_LLM=ollama:gpt-oss:120b uv run python -m fresh_replenish.agent.evaluate  # OLLAMA_API_KEY in .env (or google_genai:gemini-3.6-flash)
uv run python -m fresh_replenish.agent.chat "Place an order of 6 units of product 4 for store 57 on 2024-07-02."  # you approve hand-offs
uv run pytest                                    # data and model tests run once the data and weights are downloaded
```

## What the audit found

- **Size:** 4.85 M daily rows covering 50,000 store-product series, 898 stores and 18 cities. Train runs 90 days and eval 7 days.
- **Stockouts:** 19.7% of operating hours (6:00–22:00) are stockouts, and 44% of product-days (one product at one store for a day) have at least one.
- **Time of day:** stockouts climb from about 5% at 7 am to 41% by 9 pm. Evening demand is the most hidden, which is the problem this project goes after.

## What drives demand

These are associations, not causes.

- **Price:** a 10% price cut goes with +14.2% sales (price elasticity −1.26).
- **Non-working days:** they lift sales by 31.7%.
- **Promotions:** holding price fixed, they add +1.4%. On in-stock days alone they add +5.8%, a hint that stockouts hide promotion demand.
- **Stability:** all 7 drivers are stable across random halves of the stores.

## Recovering hidden demand

Methods were scored on in-stock hours we hid ourselves, using the dataset paper's protocol, so the true demand is known.

- **Without recovery,** demand on those days is understated by 70.9%.
- **The simple in-stock-hour profile wins,** with WAPE 37.9% and no bias. SAITS (45.5%), Tobit (47.7%) and TimesNet (71.0%) all do worse.
- **Effect:** recovered demand adds 28% to total train demand, and 62% on stockout days.

## Forecasting

Each model forecast recovered demand 7 days ahead from 4 weekly starting points, for 4,928 store-product series.

- **Every model beats seasonal naive in every fold.** Seasonal naive repeats last week; its WAPE is 44.2%.
- **LightGBM is the most accurate,** at WAPE 33.2%.
- **The zero-shot foundation models come within 1–2 points:**
  - Chronos-2 with known inputs: 34.3%;
  - TimesFM-2.5: 34.5%;
  - Toto-2.0: 35.1%.
- **Known inputs** (discount, promotion, holiday) help Chronos-2 by 0.6 points.
- **No leakage:** none of the foundation models' published training data includes this dataset.

## Calibrated ranges and cost-based orders

The checks use a held-out week (Jun 26–Jul 2) that no model saw.

- **Calibrated ranges:** split-conformal calibration per product category brings every model's 10–90% range to 78.7–80.0% coverage (target 80% ± 2), up from 74.9–78.5%.
- **Costs are assumed:** with no prices in the data, a lost sale costs 0.5× to 9× a wasted unit. The headline is 9×, because quick commerce runs availability-first.
- **Cost-based orders win:** at 9×, LightGBM's cost-based orders cost 55% less than reordering last week's demand (recovered, so its stockout gaps are filled in).
- **The most accurate number is a poor order:** ordering LightGBM's median costs 2.3× as much as its cost-based order, and more than reordering last week's demand.
- **LightGBM is cheapest overall,** at every cost ratio. A foundation model is cheapest in 27 of the 90 stores.

## Service

- **Batch job:** each night it trains LightGBM, calibrates the forecasts and writes cost-based orders for all 50,000 series. A run takes 11.5 minutes on this laptop and is tracked in MLflow.
- **API:** `GET /reorder?store_id=…&ratio=…` returns a store's orders, recomputed for any cost ratio from 1/9 to 9. It reads the orders again whenever the nightly batch rewrites them. p95 latency is 7.3 ms.
- **Monitoring:** it checks median bias, range coverage and stockout drift, overall and per category. On the held-out week everything is within limits overall (coverage 81.5%), and 13 of 32 categories raise at least one alert.
- **Docker:** one slim image (no ML stack) runs the API and the dashboard, and `docker compose up` adds MLflow. GitHub Actions builds the image and smoke-tests `/health` and `/reorder`.

## Agent

A LangGraph agent turns the forecasts into explained orders.

- **Tools:** four plain-code tools: forecast lookup, cost-based order within limits, Chronos-2 what-if, and decision record.
- **Numbers:** a checker rejects an answer with a number that isn't in a tool output or the question; small counts and ranks like "90th" are allowed.
- **Hand-off:** orders over the limits pause for a planner's approval.
- **Evaluation:** 50 scripted questions, each run three times, score tool choice, numbers, hand-offs, refusals and latency (and cost, when `FRESH_LLM_PRICE` is set).
- **Results** ([agent evaluation](reports/agent_gpt-oss-120b.md)): on Ollama Cloud `gpt-oss:120b`, re-run after the audit fixes (misses are in [negative results](docs/negative_results.md)):
  - correct tool choice 100%;
  - every final answer passed the number checker;
  - every over-limit order was handed to a planner;
  - every out-of-scope and missing-data question was handled correctly;
  - p50 latency 3.5 s.

## Data

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) by Dingdong, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the paper: [arXiv 2505.16319](https://arxiv.org/abs/2505.16319).

## Licence

Code: [MIT](LICENSE). The dataset keeps its own CC BY 4.0 licence.
