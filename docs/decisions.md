# Decisions

Each entry gives the decision, why it was made, and the evidence behind it.

## 2026-11-03: `hours_stock_status` = 1 means out of stock

- **Why:** night hours (0–2, 22–23) are flagged 1 even on days with no sales, so 1 must mean "not available".
- **Evidence:** on all 4,850,000 rows, `stock_hour6_22_cnt` equals the number of 1s in hours 6–21 (audit check `stock_count_mismatch` = 0).

## 2026-11-03: Stockout rates use the operating window, hours 6–21

- **Why:** night-hour flags would inflate stockout rates. The dataset's own count covers 6:00–22:00.
- **Evidence:**
  - 19.7% of operating hours are stockouts.
  - 44.0% of store-days have at least one stockout hour.
  - The share rises from 4.9% at 7:00 to 40.7% at 21:00, so evening demand is the most hidden.

## 2026-11-03: Pin the dataset revision and check file sizes

- **Why:** the upstream dataset was last changed on 2026-10-09, and results must be reproducible.
- **Evidence:** revision `c374e85975c82816c495acfa904cc55ab14a47d3`; files of 106,436,287 and 8,440,124 bytes. The download skips a file with the right size and re-fetches a short one.

## 2026-11-03: Query with DuckDB views, not pandas

- **Why:** 4.85 M rows carry two 24-value list columns, and the laptop has 13.9 GB of RAM.
- **Evidence:** the full audit runs in about 22 s straight from the Parquet files.
- **Gotchas found** (DuckDB 2.0):
  - a connection is blocked while a result is half-read;
  - relations no longer have `fetchone()`.

  So results are always read in full, with `fetchall()` or `.df()`.

## 2026-11-03: Data stays out of git

- **Why:** the dataset is 115 MB and licensed CC BY 4.0. Anyone can fetch the same files with the pinned download command.

## Facts that differ from the paper

- **Products:** the data has 865 distinct products (paper: 863; dataset card: 865). We use what the data says.

## Open questions for Phase 2

- **Discount values:** `discount` ranges from 0.0 to 1.088, and 51.4% of rows are below 1. It looks like a price ratio. Rows at 0.0 and above 1 need checking before the demand-driver regression.
- **City balance:** city 0 holds 2,503,667 rows (51.6% of all rows). Analyses should report per city, or weight cities, so one city doesn't dominate.
