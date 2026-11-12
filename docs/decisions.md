# Decisions

Each entry gives the decision, why it was made, and the evidence behind it.

## 2026-11-03: `hours_stock_status` = 1 means out of stock

- **Why:** night hours (0–2, 22–23) are flagged 1 even on days with no sales, so 1 must mean "not available".
- **Evidence:** on all 4,850,000 rows, `stock_hour6_22_cnt` equals the number of 1s in hours 6–21 (audit check `stock_count_mismatch` = 0).

## 2026-11-03: Stockout rates use the operating window, hours 6–21

- **Why:** night-hour flags would inflate stockout rates. The dataset's own count covers 6:00–22:00.
- **Evidence:**
  - 19.7% of operating hours are stockouts.
  - 44.0% of product-days (one product at one store) have at least one stockout hour.
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
  | Seasonal naive (last week's recovered demand) | 2.111 | 78.8% | 48.3% | 20.7% |
  | LightGBM median (the most accurate number) | 2.166 | 77.0% | 56.5% | 11.0% |

- **The decision rule matters more than the model.** At r = 9, ordering LightGBM's median costs 2.3× its cost-based order, and even more than seasonal naive. The gap appears whenever a lost sale costs more than a wasted unit (r ≥ 2), and grows with r.
- **Across models (negative result for the spec's claim):** LightGBM's cost-based orders are the cheapest at every ratio. Every other model's paired interval is above 0: +0.003 to +0.12 per unit of demand.
- **Store level:** at r = 9, a foundation model is cheapest in 27 of the 90 stores. That per-store comparison has no intervals, and 7 days per store is a small sample.
- **Cost of availability:** a 95.7% fill rate means about a third of the units ordered are wasted. That is the price of r = 9 with forecasts this uncertain; lower ratios trade the other way.

## 2026-11-03: Service design (spec step 7)

- **Batch job:** it runs "as of" the last train day (Jun 25) and orders for the held-out week, so monitoring can check it against what happened.
  1. Trains LightGBM, the Phase 4–5 winner, on recovered demand for all 50,000 series.
  2. Adds the week's known inputs.
  3. Calibrates the deciles with Phase 5's conformal shifts. Categories not seen in calibration use the pooled shift.
  4. Orders at the 9× headline ratio.
- **API:** serves the precomputed orders, one store per request.
  - Another ratio is recomputed from the stored deciles, so no model runs at request time.
  - Errors: 404 for an unknown store, 422 for a ratio outside 1/9–9, 503 before any batch has run.
  - It is built by a factory (`uvicorn --factory`), so importing it reads no data.
- **MLflow:** full `mlflow` with a SQLite store (`mlruns/mlflow.db`). MLflow 3.17 refuses the old file store, and `mlflow-skinny` can't write to SQLite.
- **Dependency groups:**
  - `ml` holds the models and MLflow;
  - `serve` holds FastAPI and Uvicorn;
  - all groups are installed by default, so `uv sync` and CI are unchanged.
  - The Docker image installs only `serve`: 33 packages, no torch. To make that possible, the API imports nothing heavy.
- **Docker:** checked in CI only, per the roadmap option, so Docker Desktop isn't installed. The image job runs on GitHub once the repo is pushed; until then it is unverified.
  - Locally, an environment synced with only the `serve` group runs the API against the real orders file.
  - The Compose file adds the MLflow UI (`ghcr.io/mlflow/mlflow:v3.17.2`, tag checked on GHCR).
- **Monitoring alerts:**
  - median bias outside −25% to +5% (the median normally runs 10–14% low);
  - coverage outside 75–85%;
  - stockout-day rate more than 5 points above the 4 weeks before.

  Groups under 70 series-days never alert, because tiny categories gave noise alerts.

## 2026-11-03: Service measurements

- **Batch:** 50,000 series and 350,000 order lines in 11.5 minutes on this laptop (CPU LightGBM, 9 quantile models).
- **API:** p95 latency of 7.3 ms per store request (p50 5.1 ms), over HTTP on this laptop, 500 requests one at a time.
- **Monitoring of the held-out week (all series):**
  - Overall: median bias −10.4%, coverage 81.5%, stockout days 41.0% against 41.2% the 4 weeks before. No overall alert.
  - Category alerts:
    - bias in 4 categories, for example category 28 at +14.7%, where stockout days fell from 22% to 4%;
    - coverage in 5 categories;
    - stockouts in 8 categories, for example category 14, from 20% to 52%.

## 2026-11-03: Replenishment agent design (spec step 8)

- **Graph:** one LangGraph agent with four plain-code tools (`get_forecast`, `compute_order`, `what_if`, `write_decision_record`) over the nightly batch's calibrated forecasts. The LLM is chosen by `FRESH_LLM` through LangChain's `init_chat_model`, so the provider is a setting.
- **Assumed limits** (the data has no prices or capacities):
  - an order above 1.5 × the item's calibrated 90th-percentile forecast needs a planner;
  - so does a store-day total above 400 units, against a planned 99th percentile of 366.
- **Hand-off:** the graph pauses (a LangGraph interrupt) for every over-limit order, and after 3 failed limit checks since the last review.
  - `write_decision_record` checks the limits itself, because a tool call in the same step can't see the others. It records an over-limit order only if a planner approved that exact line.
- **Number checker:** every number in an answer must appear in a tool output, the question or the agent's rules.
  - It allows rounding, a fraction stated as a percentage, and counts 0–10.
  - A rejected answer gets 2 retries, then becomes a refusal.
- **What-ifs:** Chronos-2 forecasts the planned and the changed inputs in one batch, and the answer is labelled a model-based estimate.
- **Evaluation:**
  - 50 questions in `eval/questions.jsonl`, drawn from the real orders with seed 42: 10 lookups, 8 "why this order?", 6 other cost ratios, 8 what-ifs, 8 limit violations, 4 records, 4 missing-data and 2 out-of-scope.
  - Each runs 3 times, and a planner rejects every hand-off.
  - Measured: tool choice, numbers matching tools (final and first try), hand-offs, refusals, latency, tokens, and cost when `FRESH_LLM_PRICE` is set.
- **Status:** built and tested with a scripted model (26 tests). The real evaluation waits for the LLM choice.

## 2026-11-03: LLM providers: Ollama Cloud (gpt-oss:120b) and Gemini 3.6 Flash

- **Why:** the user chose Gemini and Ollama Cloud (no Groq). Keys come from the environment or a git-ignored `.env`.
- **Gemini:**
  - 3.8 Flash returned 503 errors ("high demand"), and 2.5 Flash is closed to new users, so we use 3.6 Flash.
  - The key is on the free tier: 20 requests per day per model, while the 150-run evaluation needs about 375. A Google AI Plus subscription covers the Gemini app, not API keys.
  - The Gemini evaluation waits for billing on the key's project.
- **Ollama Cloud:** `gpt-oss:120b`, called on `https://ollama.com` with `OLLAMA_API_KEY`.
- **HTTPS:** the laptop's antivirus inspects HTTPS with its own root certificate. `truststore` makes Python use the Windows certificate store instead, so verification stays on.
- **Fixes found with the real model:**
  - gpt-oss writes typographic dashes and spaces (`2024‑06‑29` with non-breaking hyphens), and the checker now reads them as plain ones.
  - The agent's rules name the median as the 50th percentile.
  - A written quantity like `1.0` must come from a tool; only whole counts 0–10 are exempt.
  - Questions other than limit questions avoid store-days already over capacity: one record question fell on a 424-unit day, where the agent's hand-off was correct.
- **Provider errors:** after 3 in a row (a quota or an outage), the evaluation stops and keeps the runs that finished.

## 2026-11-03: Agent evaluation on Ollama Cloud gpt-oss:120b meets every target

- **Test:** 50 scripted questions × 3 runs = 150 runs, 0 provider errors, 9 minutes. A planner rejects every hand-off.

  | Measure | Result | Target |
  |---|---|---|
  | Correct tool choice | 97% (146 of 150) | ≥ 90% |
  | Numbers matching tool outputs, final answers | 100% | 100% |
  | Passed the number checker on the first try | 96% | — |
  | Hand-off on limit violations | 100% (24 of 24) | 100% |
  | Refusals and missing data handled | 100% (18 of 18) | — |
  | Latency per question | p50 3.3 s, p95 6.1 s | — |
  | Tokens per run | 2,419 | — |

- **Misses** (`reports/tables/agent_runs_gpt-oss-120b.csv` keeps every answer):
  - in 3 runs the agent computed an order but didn't record it when asked;
  - in 1 run it recorded an order nobody asked for.
- **Not reported:** cost per question. Ollama Cloud is a subscription, with no price per token. The Gemini run, which would report a cost, waits for billing.
- **First run:** it scored 52% on numbers, which turned out to be a scoring bug: the scorer checked answers without the agent's rules text, which defines "10th/50th/90th percentile". The scorer now uses the same sources as the checker.
  - The same run showed that "isn’t available" wasn't recognised as a missing-data answer.
  - The record rule also read as a permission, not an instruction.
  - All three are fixed, and the second run is the one reported.

## 2026-11-03: Communication (spec step 9)

- **Headline:** ordering by cost cut simulated waste + lost-sales cost by 56% compared with ordering the most accurate forecast's median (55% against reordering last week's demand). The cost ratio is assumed at 9×, and the comparison is held-out week, LightGBM, 4,928 series. It leads the README, the dashboard and the one-page summary, always with the cost assumption beside it.
- **Dashboard:**
  - Built with Streamlit, plus Altair, which ships with it; no Plotly dependency.
  - Colours are the reference palette's first three categorical slots (blue, orange, aqua), validated for colour-blind separation as a set, on a light theme pinned in `.streamlit/config.toml`.
  - It was checked by rendering it in a browser.
  - Tabs: where money is lost (by category, store and hour), ordering policies across cost ratios, forecasts, orders, and agent decision records.
- **Image:** installs the `serve` and `dashboard` groups (50 packages, no torch). Compose runs the API (:8000), the dashboard (:8501) and MLflow (:5000).
- **Docs:** `docs/summary.md` (one page, non-technical), `docs/negative_results.md`, and `docs/demo.md` (a 3–5 minute script). The demo uses `fresh_replenish.agent.chat`, where you act as the planner.

## 2026-11-03: Audit fixes (the 7 must-fix findings)

A full re-check (tests on a clean copy, re-runs of Phases 1, 2 and 5, the API, the dashboard and the demo commands) found these; all are fixed.
- **Hand-offs:** every order that fails its limits now goes to the planner, not only the latest one. After 3 failed checks, one hand-off still covers them all, and it shows a failed order.
- **Store-day cap:** saved orders now count toward the 400 units, including orders saved in earlier sessions. The check and the save run under one lock, because parallel tool calls run in threads. Each evaluation run now starts with an empty record file, so earlier runs can't change its limit checks.
- **Chat output:** the chat command writes UTF-8, so the model's typographic dashes no longer crash it when output is piped on Windows.
- **Ollama timeout:** Ollama Cloud requests now time out after 120 s, like Gemini's; demo runs had hung for minutes.
- **Baseline label:** the seasonal-naive policy reorders last week's *recovered* demand, not sales, and every label now says so. Reordering last week's recorded sales would cost more, so the comparison as built is the conservative one.
- **Demo and summary:** moving the slider from 9 to 1 shifts cost from waste to lost sales; the median costs more than the naive policy only at 9×; it runs out on 56% of days, not 57%; the simulation uses recovered demand; the what-if example now uses a day without a planned promotion.
- **CI image check:** curl retries any connection error while uvicorn starts, and the smoke test serves a one-line orders file through `/reorder`.

## 2026-11-03: Audit fixes (the 6 should-fix findings)

- **Number checker:** the agent's rules no longer count as a source, so their 9, 50 or 90 can't pass as an order quantity; ranks like "90th" are allowed instead. A fraction counts as a percentage only when a % sign follows it, and numbers written as words above ten ("twelve") are checked.
  - Still not read: numbers stuck to letters ("Rs500"). The data has no prices, and reading them would reject identifiers like "q90".
  - Dates written in words are still rejected; the agent then retries with YYYY-MM-DD.
- **Negative orders:** `compute_order` refuses a negative or NaN quantity.
- **API:** it reads `orders.parquet` again whenever the file changes, so a new nightly batch needs no restart. `/health` returns 503 until orders are loaded, so container health checks show it.
- **Dashboard health check:** Compose gives the dashboard its own check on Streamlit's `/_stcore/health`.
- **Wording:** the README and summary now say store-product lines (538 products), product-days for the 44% stockout share, 50 questions × 3 runs, ratios from 1/9 to 9, 13 of 32 alerting categories and the current Docker setup. The agent's number claim now says what was measured: every final answer passed the number checker.
- **Evaluation:** negative results now say the agent scores are in-sample and were measured before the audit fixes. Re-run on 2026-11-04 with the current code, after offering the save tool only when the question asks to place, record or save an order: tool choice, hand-offs and refusals 100%, all 150 final answers pass the number checker (79% on the first try), p50 3.5 s.

## 2026-11-04: Audit fixes (the 6 nice-to-have findings)

- **Ratio message:** the agent's error now says "between 1/9 and 9"; it said 0.11, which itself fails the check.
- **Lowest ratio:** `order_at` keeps its interpolation weight between 0 and 1, so the 1/9 ratio can't order a hair below q10.
- **Audit:** a missing hourly list now fails `bad_hour_lists` (its NULL length used to be skipped), and the stock-count check is NULL-safe. The pinned files still pass.
- **Refusal:** when the agent gives up after rejected answers, it says if an order from the conversation was already saved.
- **Dashboard:** the orders file is read with the other pipeline files, so a missing batch shows the "run the pipeline" message instead of a traceback.
- **Negative results:** the wrong-sign what-if example now points to a saved run (evaluation question 30).

## 2026-11-05: A second delivery

- **Why:** orders so far commit the whole day before opening, and stockouts climb to 41% by 9 pm. The hourly data can show what a midday top-up is worth.
- **Policy:** the morning carries a share of the one-delivery order. At the decision hour the store counts its shelf and orders up to a rest-of-day target, delivered 2 hours later.
- **Target:** the day's calibrated deciles, scaled to each line's usual share of demand still to come, then calibrated again with Phase 5's shifts on the past folds. Shares use only days before the scored days.
- **Choosing the plan:** the decision hour and morning share are picked on the 28 fold days. The held-out week only scores them.
- **A shelf count, not a forecast update:** a throwaway run on the folds also updated the forecast from the morning's pace. At 9× it moved the cost only from 0.721 to 0.689, and part of that is circular: on stockout days, the recovered afternoon is computed from that same morning. It was left out, which removes a parameter and any modelling of sold-out mornings.
- **Night sales** (4.6% of sales) count as the last hour's, so a midday count can't see them. That is the cautious choice for the top-up.
- **Result at 9×:** order at 13:00 for 15:00, with 80% sent in the morning: 0.646 per unit of demand against 0.947, a 32% saving (paired 95% CI 31–33%). On days with no stockout the saving is 23%.
- **Left out on purpose:**
  - the delivery's own cost, which the data can't give, so the report states a break-even of 24.3 wasted units per store per day at 9× (6.0 at 1×);
  - a live top-up endpoint, which would need real stock counts (added later the same day; see Live top-up below).

## 2026-11-05: Live top-up

- **Why:** the second-delivery study showed the saving but no store could use it. `POST /topup` turns a midday shelf count into units to order.
- **The batch does the work:** it adds each line's rest-of-day deciles, the plan's hours and its morning share to `orders.parquet`, using the shifts `decisions.topup` saves in `topup_shifts.csv`. The API only subtracts the shelf from a target, so no model runs at request time and the slim image stays slim.
- **Run order:** `decisions.topup` now runs before the batch, which fails with a message if the plan is missing.
- **One plan:** every store gets the plan chosen at 9× (13:00, 80% in the morning). `/topup` accepts any cost ratio for the target, but the hour and share stay fixed. The study's best differs slightly at low ratios (12:00 at 0.5×, 90% in the morning at 1× or below).
- **Counts:**
  - a store can count some lines only, and only those come back, so no line is assumed empty;
  - a product with no order that day is refused by name;
  - negative counts and NaN are refused.
- **Bug found by a test:** FastAPI refused a NaN count but then failed with a 500 trying to echo it back as JSON. Validation errors no longer echo the input.
- **Replay:** monitoring replays the served targets on the held-out week. Across all 50,000 lines in 898 stores, cost falls 31% (0.953 to 0.654; 95% CI 31–32%), and 23% on days with no stockout, in line with the 90-store study. A test checks that the served path reproduces the study exactly on the same data.
- **Daily orders are unchanged:** every decile in the new batch matches the old file exactly.
- **Timings moved, mostly from the laptop:**
  - The batch took 13.3 minutes (it was 11.5). The two new steps take 17 seconds of that, so nearly all of the rise is not the new code.
  - `/reorder` p95 is now 14.7 ms (it was 7.3). The same day, the code on main measured 19–30 ms against 15–16 ms for this branch, and the old and new orders files were level. So the rise comes from a busier laptop, not the new columns.

## 2026-11-05: Shelf life

- **Why:** every result so far throws unsold stock away the same night. The data has no shelf lives, and this is the biggest assumption behind the waste figures.
- **Simulation:** `serve` now runs day by day with stock by age. The oldest units sell first and units are thrown away after `life` days. At one day it reproduces every earlier number exactly.
- **A clean start:** a shelf that starts empty and ends full would flatter stock that keeps, so the week runs twice and the second pass is scored. Day 7 leads into day 1, which keeps the weekdays in order.
- **Assumed:** 1, 2 and 3 days, the same for every product, at 9×.
- **Result:**
  - Cost-based orders cost 0.947, 0.311 and 0.158 per unit of demand at 1, 2 and 3 days.
  - They beat a fixed multiple of the median, chosen on past weeks, by 6.5%, 11% and 8.5% (95% CIs 5–8%, 9–13%, 6–11%).
  - The second delivery saves 32%, 25% and 15%, worth 24.3, 6.4 and 2.0 wasted units per store per day.
- **Two comparisons were wrong at first, for the same reason:** a policy with re-tuned levels was judged against one left untuned.
  - The top-up's plan was re-chosen but one delivery stayed at the newsvendor level, which showed 50–57% savings at 2–3 days. Past weeks now choose its level too (1.3× and 1.5×), and the morning share may go above 100%.
  - The code review caught the same flaw in the median: "78–80% better at 2–3 days" compared two untuned policies. The tuned median needs up to 2.7×, so the search now runs to 4×.
- **What this means for the headline:** the 56% is measured against ordering the median itself, and that stays true. But a fixed 1.7× the median gets within 6.5% even at one day, so most of the gain is ordering more. The README, summary and dashboard now say so next to the 56%.
- **Left alone:** cost-based orders stay at the newsvendor level at one day, where the rule holds, so their numbers match `reports/orders.md`.
- **Still not re-tuned:** the top-up's rest-of-day target stays at the critical ratio, which may understate it a little at 2–3 days.

## 2026-11-05: Compose in CI

- **Why:** `docker compose up` had never been run, because this laptop has no Docker. CI only tested the API's image. The dashboard had never run in CI at all: its test needs the dataset, so it skips there.
- **Change:** a `compose` job replaces the `image` job, since it covers the same checks. It writes tiny stand-ins for the batch's outputs, starts all three services, and checks `/reorder`, `/topup`, MLflow's `/health` and the dashboard.
- **Dashboard check:** Streamlit's health endpoint doesn't run the page, so the job opens it inside the container with Streamlit's test runner. That is not a browser.
- **Same data in the test job:** the stand-ins come from `tests/dashboard/smoke_data.py`. Tests open the dashboard and call the API on them, so a broken stand-in fails before Docker does, and the dashboard now runs in CI's test job too.
- **Guard:** the writer stops if the folder exists, so it can't replace real pipeline outputs.
- **Result:** it passed on its first run. All three services started, `/reorder` and `/topup` answered, MLflow returned its Default experiment, and the dashboard opened with its 5 tabs and no errors.
- **What the log showed:** Streamlit warned 6 times that `use_container_width` is going away. Its default already stretches these charts, so the argument is gone. All 6 still fill their space on the real data.

## 2026-11-05: Top-ups in the agent

- **Why:** the agent knew nothing about the second delivery, so a manager couldn't ask it about a midday count.
- **Tool:** `compute_top_up` works out one line the way `/topup` does, and matched it on three real lines. The answer gives the units, when the shelf is counted and when the delivery arrives.
- **Advice only:** saved as an order, a top-up would replace the day's order in the store's capacity count. So a question that names a top-up never gets the save tool, and a conversation that called `compute_top_up` can't save. Top-ups also skip the order limits, since nothing is placed.
- **No count, no answer:** the number checker doesn't read tool arguments, so an invented shelf count would pass. The prompt tells the agent to ask, and 2 of the 6 new questions test that.
- **Questions:** the 6 new ones come last, so the original 50 keep their random draws (checked line by line). That makes 56 questions and 168 runs. A new `gave_up` measure counts answers that end in "ask a planner".
- **It took five runs:**

  | | Before | 1: top-up tool | 2: rule in the prompt | 3: rule moved | 4: checker fixed | 5: published |
  |---|---|---|---|---|---|---|
  | Right tools | 100% | 99.4% | 95% | 98.8% | 96% | 99.4% |
  | Hand-off on over-limit orders | 100% | 100% | 92% | 100% | 100% | 100% |
  | Over-limit runs that saved after a rejection | 5 of 24* | 6 of 24 | 0 | 0 | 0 | 0 |
  | Passed the check first try | 79% | 77% | 78% | 80% | 98% | 99% |
  | Gave up | 9 of 150 | 9 of 168 | 8 of 168 | 11 of 168 | 0 | 0 |
  | Unasked saves on "why" and other-ratio runs | 0 | 0 | 5 | 2 | 6 | 0 |
  | Tokens per run | 2,936 | 3,309 | 6,252 | 3,392 | 9,765 | 2,383 |

  \* From tool calls; that run's records weren't kept.

- **Run 1 looked fine until the answers were read.** After a planner rejected an order, the agent placed a smaller one itself. A code check now refuses any save once a planner rejects, except an order a planner approved.
- **Run 2 added a prompt rule for the same thing, and it backfired.** The rule reached every question: 5 "why" and other-ratio runs tried to save (one made 59 tool calls), 2 over-limit runs skipped the planner, and 2 of 12 record runs never saved.
- **Run 3 moved the instruction into the message that reports a rejection,** so other questions see the same prompt as run 1. All 24 over-limit runs say no order was placed, all 12 record runs saved, all 12 top-ups with a count were right, and all 6 without one asked.
- **Run 4 fixed the give-ups.** A 14-question sample kept every answer the checker saw: all 29 rejected numbers were percentile labels typed oddly, like "90-th" or "Median (50 %)". The model wasn't inventing anything. The checker now reads those labels, and real inventions like "Demand rises 50 %" still fail. A new `rejected` column keeps what it rejects.
- **Run 4 also showed why the unasked saves happen.** The rules named the save tool on questions where it wasn't offered. The model guessed its arguments, and the error told it to fix them and try again. One run made 92 calls and used 1.03 million tokens, so the evaluation used 1.64 million instead of the 0.4 million estimated.
- **Run 5 removed that cause and capped the damage.** The save rule now appears only with the save tool, and the agent stops after 5 failed tool calls, handing any waiting order to a planner first. There were no unasked saves, no run made more than 3 calls, and the evaluation used 0.4 million tokens. 167 of 168 runs finished; one hit an Ollama Cloud server error.
- **Left alone:** the tool assumes a count at 13:00, as `/topup` does, and doesn't state the morning share. Five or more missing products end in a refusal, and loops of successful calls aren't capped. All are in the limitations.

## 2026-11-09: Held-out questions

- **Why:** the agent's prompt, scorer and checker were tuned on the same 56 questions they're scored on, over many runs. Those scores show what the agent does on questions it was fitted to, not on new ones.
- **Questions:** `eval/questions_holdout.jsonl` has the same 9 types and counts, drawn again with seed 7, so no store-product-day overlaps the tuned set. Every question is reworded ("How many units of product 333 will store 835 sell on 2024-07-01?"), and the two out-of-scope questions are new.
- **Same tools on both:** the new wording offers the save tool exactly where the tuned wording does, and a test checks it. Otherwise a reworded record question could fail only because "save" changed to "keep".
- **Rules for the run:** the questions are committed before any run. They are run once, with the agent as it is, and nothing is tuned on them afterwards. A second run would make them in-sample too, so the command refuses to run when the held-out report exists, and an unknown argument no longer falls back to the tuned questions.
- **An accidental partial run:** a new test, run before the guard it checks existed, started the held-out evaluation for real and was stopped after about 5 minutes. Its answers went to a temporary folder that nobody read, and nothing was changed because of it. The test now passes empty folders, so it can never reach a model.
- **Still in-sample:** the scorer's refusal and missing-data patterns were fitted to answers to the tuned questions, so a correct refusal worded differently could count as a miss.
- **Result (run once):**

  | | Tuned questions | Held-out questions |
  |---|---|---|
  | Runs that finished | 167 of 168 | 168 of 168 |
  | Right tools | 99.4% | 98.8% |
  | Answers passing the number check | 100% | 100% |
  | Passed it on the first try | 99.4% | 97.6% |
  | Gave up | 0 | 0 |
  | Over-limit orders sent to a planner | 24 of 24 | 23 of 24 |
  | Refusals and missing data handled | 100% | 100% |
  | Top-ups right, or a count asked for | 18 of 18 | 18 of 18 |
  | Tokens per run | 2,383 | 2,369 |

- **What it means:** the scores held up on new wording and new items, so the tuning didn't only fit the 56 questions. The new out-of-scope questions (a poem, an email) were all refused.
- **What it found:** 2 "why" answers skipped the forecast lookup. More important, 1 over-limit order never reached a planner: the agent went straight to saving, the save tool refused the amount, and no hand-off followed, because the hand-off only watches `compute_order`. Nothing was saved, but no planner was asked either. The tuned questions never showed this.
- **Fixed after the run:** a save refused for the limits now counts as a failed check, so the order goes to a planner, once, even if the agent skipped `compute_order`. Scripted-model tests cover it. Neither evaluation has measured it: the held-out questions can't be run again, and the published scores describe the agent before the fix.
- **Command:** `FRESH_LLM=ollama:gpt-oss:120b uv run python -m fresh_replenish.agent.evaluate holdout` writes `reports/agent_holdout_gpt-oss-120b.md` and its own decision records, so the tuned results stay as they are.

## 2026-11-10: Where a second delivery pays

- **Why:** the second delivery's saving (24.3 wasted units per store per day) is an average, and its own cost isn't in the data. A delivery costs about the same wherever it goes, so the useful question is which stores earn it back.
- **Method:** each store's saving under the chosen plan (13:00 count, 80% sent in the morning) is measured on the 28 past days. A store gets the delivery if that saving beats the delivery's cost. The held-out week scores the choice. The cost is tried from a quarter of the average past saving to twice it.
- **A hard rival:** choosing stores is judged against the better of delivering everywhere or nowhere. That rival is picked with hindsight on the held-out week, so the gain isn't flattered.
- **Result:**
  - The saving is uneven: the top fifth of the 90 stores hold 43% of it, and 27 save less than half the average.
  - It is predictable: a store's past and held-out savings have a rank correlation of 0.95.
  - At a delivery cost of 22.4 (the average past saving), every store nets 2.0 per store per day and the 37 chosen stores net 7.7, a gain of 5.8 (95% CI 4.4–7.1). Hindsight could reach 7.9.
  - At half that cost the gain is 1.1 (0.6–1.5), and at a quarter 0.2. At twice the cost, 9 stores still earn it back.
- **A simpler ranking does as well:** picking as many stores by past demand nets 7.6 at that cost, against 7.7. Store size carries the order, but the number of stores still comes from the simulated savings.
- **No regression to the mean seen:** the chosen stores saved 36.1 on past days and 41.2 on the held-out week, which was about 10% busier.
- **Limits:** 90 stores and one held-out week. Intervals resample stores, not days, and hold the delivery cost and the chosen stores fixed, so they are too narrow. Units are the data's normalised sales for the sampled lines. The dashboard shows it as a chart of net saving against delivery cost.

## 2026-11-11: Repeat and late shelf counts

- **Why:** `/topup` had two known gaps. A second count before the delivery landed ordered the same units again, and a count taken later than 13:00 was scored against the 13:00 target.
- **No server state:** the API still keeps no record of orders. The store's system already knows what it ordered, so the count can carry `on_order`, and the top-up becomes target minus shelf minus units on order. A stored record would need a writable volume and locking for the same result.
- **Count hour:** the count can carry `hour`. Any hour but the plan's is refused with a message naming the right one, because the batch only has targets for that hour.
- **Both are optional,** so existing callers and the CI smoke call are unchanged. A caller that sends neither is trusted as before, which is in the limitations.
- **Not done:** targets for other hours, and the same two fields on the agent's top-up tool. Changing the agent's tools would make its published scores describe an older agent.

## 2026-11-12: Package upgrades

- **What moved:** `uv lock --upgrade` moved 40 packages within the project's constraints. Among them: FastAPI 0.162.5, Streamlit 1.67, MLflow 3.18, LangChain 1.5, LangGraph 1.2.20, pyarrow 26, and two majors, pypots 2.0 and pyfixest 0.70. All tests pass, including the ones on the real data and model weights.
- **Held back:** pandas stays on 2.3 (toto-2's gluonts caps it below 3), transformers on 4.57 (pypots and chronos-forecasting), and torch on 2.14 (the project's own pin, with the CUDA 12.6 wheels).
- **MLflow migrates its store:** the 3.18 client upgrades the SQLite schema the first time it logs a run, checked on a copy of `mlruns/mlflow.db`. A 3.17 server can't read the upgraded store, so the Compose file now runs `ghcr.io/mlflow/mlflow:v3.18.0` (tag checked on GHCR). The local store is migrated by the next batch run.
- **Results not re-run:** the reports come from outputs saved with the old versions. pyfixest 0.70 and pypots 2.0 could move the Phase 2 and Phase 3 numbers slightly if those phases were run again.

## Known issues

- **Hugging Face downloads on this laptop:** the antivirus intercepts HTTPS, so Hugging Face requests fail certificate checks. The forecasting run therefore used `HF_HUB_OFFLINE=1`, with the pinned weights already in the local cache.
- **pyfixest warnings:** pyfixest 0.60 uses pandas options deprecated for pandas 4, which produces warnings. Versions are pinned in `uv.lock`.

## Facts that differ from the paper

- **Products:** the data has 865 distinct products (paper: 863; dataset card: 865). We use what the data says.

## Phase 1 open questions, now answered

- **Discount values:** `discount` is a price ratio. Rows at 0 are dropped (see above). The 34 rows above 1 are kept as small price rises.
- **City balance:** city 0 holds 51.6% of rows, so every analysis also reports a fit without it.
