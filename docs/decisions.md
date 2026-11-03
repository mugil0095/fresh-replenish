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

## 2026-11-03: Drop `discount = 0` rows from the driver regression

- **Why:** `ln(0)` is undefined, and these rows aren't normal prices. They sell 14.5× the normal mean in train and 23× in eval, so they look like giveaways or a coding quirk.
- **Evidence:** 16,039 train rows and 1,300 eval rows (0.36%). They are reported separately in `reports/demand_drivers.md`.

## 2026-11-03: Poisson regression (PPML) with store-product and week fixed effects, clustered by store

- **Why:**
  - It handles zero-sales days (4.5% of train).
  - It is unaffected by the normalised sales scale.
  - Log price gives a price elasticity directly.
  - Week fixed effects keep the date-level holiday flag measurable; date fixed effects would absorb it.
- **Evidence:** a simulation test recovers 7 known effects within 4 standard errors. The full fit on 4.48 M rows takes 33 s.

## 2026-11-03: `holiday_flag` means "non-working day"

- **Evidence:** it is the same for every store on a given date. It covers 33 of 97 days: all weekends, plus Qingming (Apr 4–6), Labour Day (May 1–5) and Dragon Boat (Jun 10).

## 2026-11-03: Demand drivers (associations, not causes)

- **Price:** elasticity −1.26, so a 10% price cut goes with +14.2% sales (95% CI +13.8% to +14.6%).
- **Non-working day:** +31.7% sales.
- **Promotion:** +1.4% holding price fixed. Weather effects are about 1% per unit or less.
- **Stability:** 7 of 7 drivers are stable across two random halves of the stores. Price is the closest call (z = −1.94 against a 1.96 bound).
- **Robustness:**
  - Without city 0, the elasticity is −1.10 (same sign, smaller).
  - On in-stock days the promotion effect is four times larger (+5.8% vs +1.4%). That fits stockouts hiding promotion demand; Phase 3 tests it.
  - Weather is fragile: wind changes sign on in-stock days.

## 2026-11-03: Hidden-demand recovery follows the dataset paper's own protocol

- **Why:** scores on known truth, comparable with the paper.
- **How:**
  - Each series' 90 train days are cut into 3 windows of 30 days × hours 6–21 (480 steps), with stockout hours left blank.
  - `mimic_missing(p=0.3)` is ported from `Dingdong-Inc/frn-50k-baseline` (Apache-2.0), with a seeded random generator.
  - Daily totals are scored only on days that got hidden hours and had no real stockout (WAPE and WPE, as in the paper's `evaluation_mnar`).
  - Intervals come from 1,000 bootstrap resamples of series, plus paired differences against the best method.

## 2026-11-03: All four methods run on the same 10% store sample

- **Why:** SAITS and TimesNet must fit the GTX 1650's 4 GB of memory, and one shared sample keeps the comparison fair.
- **Settings:**
  - SAITS: 2 layers, d_model 64, 4 heads, d_ffn 128, dropout 0.1.
  - TimesNet: 2 layers, top_k 3, d_model 32, d_ffn 64, 4 kernels.
  - Both: 10 epochs, batch 32, seed 42.
  - Covariates (discount, holiday, rain, temperature) are standardised; sales are not.

## 2026-11-03: Series that never had an in-stock day borrow the typical uplift

- **Why:** censored-only data always push the Tobit's demand estimate up, so the prior alone decided. A test showed 2.05 recovered against 2.0 observed.
- **Rule:** these series stay out of the Tobit fit and get observed × (mean in-stock day ÷ mean sold-out day) from the fitted series.

## 2026-11-03: Deep models predict a few series at a time

- **Why:** PyPOTS gathers every batch output, and SAITS returns 480 × 480 attention maps per window and head. All at once, that's about 55 GB for the sample.
- **Rule:** predict about 21 series (≈64 windows) per call and keep only the filled values.

## 2026-11-03: The in-stock-hour profile recovers demand for Phase 4

- **Test:** 90 stores (4,928 series), 12.7% of hours hidden, 76,684 days scored.

  | Method | WAPE | Bias (WPE) |
  |---|---|---|
  | No recovery | 70.9% | −70.9% |
  | **Profile** | **37.9%** (95% CI 37.2–38.6%) | **−0.0%** (CI −0.4% to +0.4%) |
  | SAITS | 45.5% | −43.3% |
  | Tobit | 47.7% | +35.9% |
  | TimesNet | 71.0% | −70.8% |

- **Choice:** the profile method has the lowest WAPE, 7.6 points better than SAITS (paired 95% CI 7.1–8.1). It is also the only unbiased method, so the selection rule picks it directly.
- **Negative results:**
  - With these settings (10 epochs), SAITS under-recovers.
  - TimesNet learned to fill hidden hours with zeros, so it is no better than doing nothing.
  - Tobit over-recovers.
  - Tuning the deep models was outside the plan.
- **Effect:** recovered demand for all 50,000 series adds 28.2% to total train demand, and 61.9% on stockout days.
- **Rule:** on a stockout day, recovered demand is never below recorded sales, because a stockout hour can still record sales. Before this rule, 50,028 days fell below.
- **Run time (GTX 1650):**
  - profile: 0.2 s;
  - Tobit: 105 s;
  - SAITS: 17 min;
  - TimesNet: 25 min;
  - whole run: 44 min.
- **Note:** after a formatting fix, the report was regenerated from the run's saved tables, and the saved recovered-demand file was patched with the rule above. A full rerun gives the same output.

## 2026-11-03: PyTorch 2.14.1 with CUDA 12.6, and a floor on `transformers`

- **Why:** CUDA 12.6 builds still support the GTX 1650 (sm_75); CI on Linux gets CPU builds.
- **Install conflict:** PyPOTS 1.5 caps `transformers` at 4.57.6. Without a floor, uv backtracked to 4.12, which needs a Rust compiler. The floor (`>=4.40`) pulls `huggingface_hub` down to 0.36, and our download call works there unchanged.

## 2026-11-03: FreshRetailNet-50K isn't in any foundation model's pretraining data

- **Why:** a model that saw these series during pretraining would score unfairly well (the spec's "a model may have seen this dataset" risk).
- **Evidence:** the dataset was first published on 2025-05-07 (Hugging Face; arXiv 2505.16319).

  | Model | Pretraining data, from its model card or report | Contains FreshRetailNet-50K? |
  |---|---|---|
  | Chronos-2 | Subsets of the Chronos datasets (last changed 2025-03-18) and GIFT-Eval Pretrain (last changed 2025-01-21), plus synthetic data | No. Both corpora predate the dataset, and neither lists it. |
  | TimesFM-2.5 | GIFT-Eval Pretrain, Wikimedia pageviews (to Nov 2023), Google Trends (to end of 2022), plus synthetic data | No |
  | Toto-2.0-313m | Datadog observability metrics and synthetic series only. Its report says it "sees no public forecasting data during pretraining". | No |

- **Not confirmed:** the exact subsets each team used; we rely on their published lists. Toto's fine-tuned 2.5B-FT model also trained on GIFT-Eval train data, but it isn't used here.
- **Sources:**
  - the model cards of `amazon/chronos-2` and `google/timesfm-2.5-200m-pytorch`;
  - the Toto 2.0 technical report (arXiv 2605.20119), sections 1 and 3;
  - the Hugging Face API dates of the two corpora.

## 2026-11-03: Forecasts are validated on 4 weekly folds inside the train period

- **Why:** each fold scores a realistic weekly forecast from all the history before it. The eval week (Jun 26–Jul 2) stays untouched for Phase 5's held-out check.
- **How:**
  - Origins at days 62, 69, 76 and 83, each forecasting 7 days, so the last fold ends on the last train day.
  - The target is Phase 3's recovered demand.
  - Scores pool the 28 forecast days per series. WAPE per fold backs the "beats seasonal naive in every fold" check.

## 2026-11-03: Forecasts use only inputs known in advance

- **Inputs:** discount, promotion (`activity_flag`) and the non-working-day flag. A retailer plans all three ahead.
- **Weather is left out:** it is known only after the fact, and Phase 2 found effects of about 1% that were fragile. Using actual weather would give the models perfect foresight that a real forecast doesn't have.

## 2026-11-03: Forecaster settings

- **Output:** every model gives deciles 0.1–0.9, clipped at 0 and sorted. The median is the point forecast. Seasonal naive gives a median only.
- **LightGBM:**
  - One global model per fold and decile.
  - Lags 7–28 and 7- and 28-day means, all ending at least 7 days back, so one model covers the whole week.
  - Day of week, the 3 inputs, and store, product, city and category ids as categorical features.
  - 200 trees, learning rate 0.05, 63 leaves, not tuned. Deterministic training, so a rerun gives the same forecasts.
- **Foundation models:** zero-shot, weights pinned to Hugging Face revisions (in `forecasting/models.py`).
  - Chronos-2 gets the inputs as past and future covariates. A second run without them shows whether they help, with the model held fixed.
  - TimesFM-2.5 uses the model card's settings: normalised inputs, continuous quantile head, no quantile crossing, positive outputs.
  - Toto-2.0-313m gets its history left-padded to whole 32-day patches, with the padding masked.
- **Sample:** the same 90 stores (4,928 series) as Phase 3. On the GTX 1650, 2,000 series take 5–20 s per model, so all 50,000 series would take about 2 hours. The sample keeps runs short, and bootstrap intervals over 4,928 series are already narrow.
- **Install:** toto-2 needs gluonts, which caps pandas below 3, so pandas moved from 3.1.0 to 2.3.3. All earlier tests pass unchanged.

## 2026-11-03: LightGBM forecasts best, and every model beats seasonal naive in every fold

- **Test:** 90 stores (4,928 series), 4 weekly folds, 137,984 forecast days per model.

  | Model | WAPE (95% CI) | Bias (WPE) | Against LightGBM (paired 95% CI) | Inside the 10–90% range |
  |---|---|---|---|---|
  | **LightGBM** | **33.2%** (32.4–34.0%) | −9.4% | — | 75.1% |
  | Chronos-2 with inputs | 34.3% (33.3–35.2%) | −8.2% | +1.0 points (+0.8 to +1.2) | 78.0% |
  | TimesFM-2.5 | 34.5% (33.6–35.5%) | −10.3% | +1.3 points (+1.1 to +1.5) | 79.2% |
  | Chronos-2 without inputs | 34.8% (33.8–35.7%) | −9.9% | +1.6 points (+1.4 to +1.8) | 79.1% |
  | Toto-2.0-313m | 35.1% (34.1–36.0%) | −9.2% | +1.9 points (+1.7 to +2.1) | 79.9% |
  | Seasonal naive | 44.2% (43.0–45.4%) | −3.5% | +11.0 points (+10.5 to +11.5) | — |

- **Exit check met:** every model beats seasonal naive in all 4 folds (fold WAPE 32–36% against 42–46%).
- **Foundation models vs LightGBM:** zero-shot, they come within 1–2 points of a LightGBM trained on these series, but don't beat it. The spec expected this could happen; it is a real result.
- **Known inputs help Chronos-2** by 0.6 points (34.8% to 34.3%). Both runs are paired against LightGBM, and their intervals don't overlap.
- **Bias:** every model's median runs 8–10% low.
  - Demand is right-skewed, so its median sits below its mean, and WAPE rewards the median.
  - Phase 5's cost-based orders use upper quantiles, not the median.
- **Coverage:** 75–80% of actual demand falls inside the 10–90% range before calibration (target 80%).
- **Run time (GTX 1650):**
  - LightGBM: 257 s (36 fits);
  - Chronos-2: 172 s, or 40 s without inputs;
  - TimesFM-2.5: 73 s;
  - Toto-2.0: 66 s;
  - whole run: 10.4 min.
- **Known limitation (code review):** Phase 3 filled stockout hours using each series' hourly profile from all 90 train days. So on stockout days, the history each fold learns from carries a little information from later weeks. This affects every model alike; recovering demand separately for each fold would remove it.

## 2026-11-03: The held-out week is scored against recovered demand

- **Why:** Phase 4 scored against recovered demand, and 42% of held-out days have a stockout, so their recorded sales understate demand.
- **How:**
  - The eval split (Jun 26–Jul 2) is recovered with Phase 3's chosen method: the in-stock-hour profile, estimated over all 97 days and never below recorded sales.
  - This covers the same 90 stores, and the result is saved to `data/processed/recovered_demand_eval_sample.parquet`.
  - Forecasts read only the 90 train days plus the week's known inputs.
- **Held-out accuracy** has the same ranking as the Phase 4 folds:
  - LightGBM: WAPE 32.5% (95% CI 31.7–33.3%);
  - Chronos-2: +0.7 points (paired CI +0.3 to +1.1);
  - TimesFM-2.5: +1.0 points;
  - Toto-2.0: +1.3 points;
  - seasonal naive: 41.7%.

## 2026-11-03: Forecast ranges are calibrated with split conformal per category

- **Why:** before calibration, the 10–90% ranges covered 74.9–78.5% of held-out demand against the 80% they claim. LightGBM's were the narrowest.
- **How:**
  - Each decile gets a one-sided shift per model and first-level category, learned on Phase 4's 4 train-period folds (137,984 forecasts per model).
  - Errors are divided by the forecast's own spread (q90 − q10), so a shift widens wide forecasts more than narrow ones.
  - The shift is the ⌈(n+1)τ⌉-th smallest scaled error.
  - Categories with fewer than 10 series share one pooled group: 6 of 29 categories, 23 series.
- **Result (exit check met):** after calibration, every model covers 78.7–80.0% of held-out demand (target 80% ± 2 points). LightGBM goes from 74.9% to 80.0%. Pinball loss also improves slightly for every model.
- **Remaining drift:** every calibrated decile sits 1–5 points below its level (q50 holds 45–48% of demand, not 50%). The held-out week runs a little above the folds' demand, and calibration can only learn from the past.

## 2026-11-03: Cost assumptions follow quick commerce's availability-first practice

- **Why:** the data has no prices. The user asked to follow the latest trend.
- **Assumption:** a wasted unit costs 1, and a lost sale costs r.
  - **Range:** r ∈ {0.5, 1, 2, 4, 9}, so orders sit at the 33rd to the 90th percentile of the forecast.
  - **Headline:** r = 9 (order the 90th percentile).
- **Evidence for the headline:**
  - Bernstein puts Blinkit's average in-stock rate at 94.4%.
  - A December 2025 newsvendor study (IEOM) found a retailer holding stockouts near 0.7%, prioritising availability over cost.
  - ECR's 2025 retail-loss study adds handling costs of 1.8% of sales on unsold food, so waste costs more than the product alone.
- **Model:**
  - one-day shelf life: unsold stock is thrown away, and missed sales are not made up later;
  - orders are interpolated between calibrated deciles.

## 2026-11-03: Cost-based orders beat ordering the most accurate number

- **Test:** 90 stores (4,928 series), 7 held-out days.
  - Cost is per unit of demand.
  - Intervals come from 1,000 bootstrap resamples of series, the same for every policy, so differences are paired.

  | Policy at r = 9 | Cost (95% CI) | Fill rate | Stockout days | Waste share of units ordered |
  |---|---|---|---|---|
  | **LightGBM, cost-based** | **0.947** (0.914–0.980) | 95.7% | 11.7% | 36.8% |
  | Chronos-2, cost-based | 1.019 | 95.1% | 12.5% | 37.8% |
  | TimesFM-2.5, cost-based | 1.041 | 95.3% | 11.0% | 39.4% |
  | Toto-2.0, cost-based | 1.068 | 95.4% | 11.4% | 40.6% |
  | Seasonal naive (last week's sales) | 2.111 | 78.8% | 48.3% | 20.7% |
  | LightGBM median (the most accurate number) | 2.166 | 77.0% | 56.5% | 11.0% |

- **The decision rule matters more than the model.** At r = 9, ordering LightGBM's median costs 2.3× its cost-based order, and even more than seasonal naive. The gap appears whenever a lost sale costs more than a wasted unit (r ≥ 2), and grows with r.
- **Across models (negative result for the spec's claim):** LightGBM's cost-based orders are the cheapest at every ratio. Every other model's paired interval is above 0: +0.003 to +0.12 per unit of demand.
- **Store level:** at r = 9, a foundation model is cheapest in 27 of the 90 stores. That per-store comparison has no intervals, and 7 days per store is a small sample.
- **Cost of availability:** a 95.7% fill rate means about a third of the units ordered are wasted. That is the price of r = 9 with forecasts this uncertain; lower ratios trade the other way.

## Known issues

- **Hugging Face downloads on this laptop:** the antivirus intercepts HTTPS, so Hugging Face requests fail certificate checks. The forecasting run therefore used `HF_HUB_OFFLINE=1`, with the pinned weights already in the local cache.
- **pyfixest warnings:** pyfixest 0.60 uses pandas options deprecated for pandas 4, which produces warnings. Versions are pinned in `uv.lock`.

## Facts that differ from the paper

- **Products:** the data has 865 distinct products (paper: 863; dataset card: 865). We use what the data says.

## Phase 1 open questions, now answered

- **Discount values:** `discount` is a price ratio. Rows at 0 are dropped (see above). The 34 rows above 1 are kept as small price rises.
- **City balance:** city 0 holds 51.6% of rows, so every analysis also reports a fit without it.
