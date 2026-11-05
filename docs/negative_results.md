# Negative results

What didn't work, or didn't hold, and is reported rather than hidden. The evidence is in [decisions](decisions.md) and the reports.

## Hidden-demand recovery (Phase 3)
- **The deep imputers lost to a simple in-stock-hour profile:**
  - profile: WAPE 37.9%, no bias;
  - SAITS: 45.5%, under-recovers;
  - Tobit: 47.7%, over-recovers by +35.9%;
  - TimesNet: learned to fill hidden hours with zeros (71.0%), no better than doing nothing.
- **Look-ahead in the recovered history:** stockout hours were filled using each series' profile from all 90 days, so the forecast folds' history carries a little information from later weeks. It affects every model alike.

## Forecasting (Phase 4)
- **The foundation models did not beat LightGBM.** Zero-shot, Chronos-2, TimesFM-2.5 and Toto-2.0 trailed by 1.0–1.9 WAPE points, with paired intervals above 0.
- **Every model's median runs 8–13% low.** Demand is right-skewed, so the median sits below the mean, and WAPE rewards the median.

## Calibration and orders (Phase 5)
- **The spec's claim fails across models.** "The most accurate model is not the cheapest" didn't hold: LightGBM's cost-based orders were the cheapest at every cost ratio. It holds only for the decision rule (ordering the median vs ordering by cost), and in 27 of 90 stores, which is too few days per store to test.
- **Calibrated deciles still run 1–5 points low** on the held-out week, whose demand was higher than the weeks the calibration learned from.
- **Availability costs waste:** at the 9× headline ratio, about a third of the units ordered are wasted.

## Second delivery
- **Updating the forecast from the morning's sales barely helped.** In a throwaway run on the past folds at 9× (not kept in the repo), it moved the cost from 0.721 to 0.689. Part of that is circular, so it was dropped. Nearly all the gain comes from counting the shelf.
- **The saving shrinks on observed days:** 32% overall, 23% on days with no stockout. Part of the headline may rest on recovered afternoon demand, and those days are also quieter, so the two effects can't be separated.
- **Waste stays high:** even with a top-up, 0.418 units are wasted per unit of demand at 9×.

## Service (Phase 6)
- **The Docker image is unverified.** It was checked in CI only, not built locally, and CI runs once the repo is pushed.
- **Category alerts are noisy:** monitoring flags bias, coverage or stockout drift in several categories even in a normal week.

## Agent (Phase 7)
- **Record misses, fixed:** after the audit fixes, 5 of 12 "record this order" runs computed the order but never saved it. Stronger wording alone overshot (14 unasked saves), so the save tool is now offered only when the question asks to place, record or save an order; the next run scored 100% on tool choice.
- **The stricter checker costs retries:** 79% of answers pass the number checker on the first try (96% under the old, looser checker); after retries, all 150 pass.
- **What-ifs are not causal.** Chronos-2's covariate effects can have the "wrong" sign: in evaluation question 30, a promotion lowered product 766's median at store 343 from 1.27 to 1.15 units (`reports/tables/agent_runs_gpt-oss-120b.csv`).
- **Gemini was not evaluated.** The free tier allows 20 requests a day per model, and the run needs about 375. A Google AI Plus subscription doesn't cover API keys.
- **The first evaluation run had a scoring bug:** the scorer rejected rule words like "90th percentile", which made "numbers matching" 52%. The fix counted the rules as a source; the reported run is the second.
- **The scores are in-sample.** After the first run, the prompt, the scorer and one question were changed, and the same 50 questions were run again. There is no fresh question set.