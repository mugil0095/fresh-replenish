# fresh-replenish

Fresh-food replenishment for quick-commerce dark stores. The project:

1. recovers the demand that stockouts hide;
2. forecasts it with time-series foundation models;
3. turns forecasts into cost-based daily orders;
4. explains each order with an AI agent.

**Status:** Phases 1–3 are done.
- **Data foundation:** the data is verified; see the [data audit](reports/data_audit.md).
- **Demand drivers:** see [what drives demand](reports/demand_drivers.md).
- **Hidden demand:** see [recovering hidden demand](reports/demand_recovery.md).

Each choice is explained in [decisions](docs/decisions.md).

## Run it

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m fresh_replenish.data.download   # 115 MB, pinned dataset revision
uv run python -m fresh_replenish.data.audit      # writes reports/data_audit.md
uv run python -m fresh_replenish.analysis.drivers  # writes reports/demand_drivers.md (~1.5 min)
uv run python -m fresh_replenish.imputation.recovery  # writes reports/demand_recovery.md (~45 min on a GTX 1650)
uv run pytest                                    # data tests run once the data is downloaded
```

## What the audit found

- **Size:** 4.85 M daily rows covering 50,000 store-product series, 898 stores and 18 cities. Train runs 90 days and eval 7 days.
- **Stockouts:** 19.7% of operating hours (6:00–22:00) are stockouts, and 44% of store-days have at least one.
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

## Data

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) by Dingdong, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the paper: [arXiv 2505.16319](https://arxiv.org/abs/2505.16319).
