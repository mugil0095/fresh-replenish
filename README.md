# fresh-replenish

Fresh-food replenishment for quick-commerce dark stores. The project:

1. recovers the demand that stockouts hide;
2. forecasts it with time-series foundation models;
3. turns forecasts into cost-based daily orders;
4. explains each order with an AI agent.

**Status:** Phase 1 (data foundation) is done. The data is verified; see the [data audit](reports/data_audit.md) and [decisions](docs/decisions.md).

## Run it

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m fresh_replenish.data.download   # 115 MB, pinned dataset revision
uv run python -m fresh_replenish.data.audit      # writes reports/data_audit.md
uv run pytest                                    # data tests run once the data is downloaded
```

## What the audit found

- **Size:** 4.85 M daily rows covering 50,000 store-product series, 898 stores and 18 cities. Train runs 90 days and eval 7 days.
- **Stockouts:** 19.7% of operating hours (6:00–22:00) are stockouts, and 44% of store-days have at least one.
- **Time of day:** stockouts climb from about 5% at 7 am to 41% by 9 pm. Evening demand is the most hidden, which is the problem this project goes after.

## Data

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) by Dingdong, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the paper: [arXiv 2505.16319](https://arxiv.org/abs/2505.16319).
