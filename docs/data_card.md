# Data card: FreshRetailNet-50K

- **Source:** [Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), pinned at revision `c374e85975c82816c495acfa904cc55ab14a47d3` (`data/train.parquet` 106,436,287 bytes, `data/eval.parquet` 8,440,124 bytes). Paper: [arXiv 2505.16319](https://arxiv.org/abs/2505.16319).
- **Licence:** CC BY 4.0. This repo downloads the files (`fresh_replenish.data.download`) and never commits them.
- **What it is:** daily sales of fresh products in the dark stores of Dingdong, a Chinese quick-commerce retailer, with hourly sales and hourly stock status.

## Size

| split | rows | series | stores | cities | products | days |
|---|---|---|---|---|---|---|
| train | 4,500,000 | 50,000 | 898 | 18 | 865 | 90 (2024-03-28 to 2024-06-25) |
| eval | 350,000 | 50,000 | 898 | 18 | 865 | 7 (2024-06-26 to 2024-07-02) |

A series is one product at one store. The data audit ([report](../reports/data_audit.md)) checks these counts, unique keys and every hourly list on each run.

## Columns

| column | meaning |
|---|---|
| `city_id`, `store_id`, `management_group_id`, `first/second/third_category_id`, `product_id` | anonymised IDs |
| `dt` | date (YYYY-MM-DD) |
| `sale_amount` | daily sales; equals the sum of `hours_sale` |
| `hours_sale` | 24 hourly sales |
| `hours_stock_status` | 24 hourly flags; 1 means out of stock |
| `stock_hour6_22_cnt` | out-of-stock hours between 6:00 and 22:00 |
| `discount` | price as a share of normal (0.8 means 20% off) |
| `holiday_flag`, `activity_flag` | holiday; promotion |
| `precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level` | daily weather |

## Things to know

- **Sales are normalised,** not units or money, and there are no prices. Every cost in this project is relative (a wasted unit costs 1, a lost sale costs r).
- **Sales understate demand:** a sold-out item records only what was on the shelf. 19.7% of operating hours are stockouts. Phase 3 recovers the hidden demand, and every later step uses recovered demand, including as the "actual" in the order simulation.
- **Night hours show as out of stock,** so stockout rates use the 6:00-22:00 window only.
- **Weather is observed, not forecast.** It is used to explain demand (Phase 2) but left out of the forecasting models.
- **Discount:** 0 values are dropped as invalid; 34 values above 1 are kept as small price rises.
- **Products:** the data has 865 distinct products; the paper says 863.
- **Balance:** city 0 holds 51.6% of rows, so analyses also report a fit without it.
