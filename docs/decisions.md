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

## Known issue

- **pyfixest warnings:** pyfixest 0.60 uses pandas options deprecated for pandas 4, which produces warnings. Versions are pinned in `uv.lock`.

## Facts that differ from the paper

- **Products:** the data has 865 distinct products (paper: 863; dataset card: 865). We use what the data says.

## Phase 1 open questions, now answered

- **Discount values:** `discount` is a price ratio. Rows at 0 are dropped (see above). The 34 rows above 1 are kept as small price rises.
- **City balance:** city 0 holds 51.6% of rows, so every analysis also reports a fit without it.
